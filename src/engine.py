from collections import namedtuple
import math
import numpy as np

from geometry import *
#from geometry_c import *

X = 0
Y = 1
Z = 2
W = 2
H = 3

TYPE_FLOOR = 0
TYPE_WALL = 1

MATRIX_ROT_YZ_XY = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1]
    ])

MATRIX_ROT_XZ_XY = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, 0, 1]
    ])

Clip = namedtuple('Clip', [
    'type',
    'points',
    'bounds',
    'data'
    ])

def project_points(pts, camera):
    #x
    vectors = pts[:,0:2] - camera.pos
    ndir = camera.near_dir
    nplane = camera.near_plane
    projs = ndir * (np.dot(vectors, ndir) / np.dot(ndir, ndir))[:,None]
    proj_lens = np.linalg.norm(projs, axis=1)
    rejs = vectors - projs
    #rej_lens = np.linalg.norm(rejs, axis=1)
    #scaled_rej_lens = camera.near / proj_lens[:,None] * rej_lens[:,None]
    #scaled_rejs = scaled_rej_lens * rejs / rej_lens[:,None]
    scaled_rejs = rejs * (camera.near / proj_lens[:,None])
    x_signs = np.sign(np.dot(scaled_rejs + nplane, nplane))
    xs = x_signs * np.linalg.norm(scaled_rejs + nplane, axis=1)

    #y
    vectors = np.empty((len(pts), 2))
    vectors[:,0] = proj_lens
    vectors[:,1] = camera.z - pts[:,2]
    ndir = np.array([camera.near, 0])
    nplane = np.array([0, norm2(camera.near_plane)])
    projs = ndir * (np.dot(vectors, ndir) / np.dot(ndir, ndir))[:,None]
    proj_lens = np.linalg.norm(projs, axis=1)
    rejs = vectors - projs
    #rej_lens = np.linalg.norm(rejs, axis=1)
    #scaled_rej_lens = camera.near / proj_lens[:,None] * rej_lens[:,None]
    #scaled_rejs = scaled_rej_lens * rejs / rej_lens[:,None]
    scaled_rejs = rejs * (camera.near / proj_lens[:,None])
    y_signs = np.sign(np.dot(scaled_rejs + nplane, nplane))
    ys = y_signs * np.linalg.norm(scaled_rejs + nplane, axis=1)

    #z
    vectors = pts[:,0:2] - camera.pos
    ndir = camera.near_dir
    projs = ndir * (np.dot(vectors, ndir) / np.dot(ndir, ndir))[:,None]
    zs = np.linalg.norm(projs, axis=1) / (camera.far - camera.near)

    return np.column_stack([
        xs * (camera.proj_width / camera.near),
        ys * (camera.proj_height / camera.near) * camera.aspect - camera.horizon_y,
        zs])

def project_point(pt, camera):
    #x
    vector = pt[0:2] - camera.pos
    if abs(1 - np.dot(vector, camera.near_dir) / (norm2(vector) * camera.near)) < 1e-10:
        proj_len = norm2(vector)
        x = norm2(camera.near_plane)
    else:
        proj = camera.near_dir * (np.dot(vector, camera.near_dir) / np.dot(camera.near_dir, camera.near_dir))
        rej = vector - proj
        proj_len = norm2(proj)
        rej_len = norm2(rej)
        scaled_rej_len = camera.near / proj_len * rej_len
        scaled_rej = rej / rej_len * scaled_rej_len
        x_sign = np.sign(np.dot((camera.near_plane + scaled_rej), camera.near_plane))
        x = x_sign * norm2(camera.near_plane + scaled_rej)

    #y
    vector = np.array([proj_len, camera.z - pt[Z]])
    ndir = np.array([camera.near, 0])
    nplane = np.array([0, norm2(camera.near_plane)])
    if abs(1 - np.dot(vector, ndir) / (norm2(vector) * camera.near)) < 1e-10:
        y = norm2(nplane)
    else:
        proj = ndir * (np.dot(vector, ndir) / np.dot(ndir, ndir))
        rej = vector - proj
        proj_len = norm2(proj)
        rej_len = norm2(rej)
        scaled_rej_len = norm2(ndir) / proj_len * rej_len
        scaled_rej = rej / rej_len * scaled_rej_len
        y_sign = np.sign(np.dot((nplane + scaled_rej), nplane))
        y = y_sign * norm2(nplane + scaled_rej)

    #z
    vector = pt[0:2] - camera.pos
    proj = camera.near_dir * (np.dot(vector, camera.near_dir) / np.dot(camera.near_dir, camera.near_dir))
    z = norm2(proj) / (camera.far - camera.near)

    return [x * (camera.proj_width / camera.near),
            y * (camera.proj_height / camera.near) * camera.aspect - camera.horizon_y,
            z]

def normalize_points(pts, camera):
    pts[:,0:2] /= [camera.proj_width, camera.proj_height]
    pts[:,0:2] -= 0.5
    pts[:,0:2] *= 2

def normalize_geoms(geoms, camera):
    geoms[:,:,0:2] /= [camera.proj_width, camera.proj_height]
    geoms[:,:,0:2] -= 0.5
    geoms[:,:,0:2] *= 2

def clip_floor(rect, camera):
    tl = rect[0]
    tr = rect[1]
    br = rect[2]
    bl = rect[3]
    edges = [[tl, bl], [tr, br], [tl, tr], [bl, br]]
    cam_l_ray = camera.rays[0]
    cam_r_ray = camera.rays[-1]
    near_clip_l = camera.pos + camera.near_dir - camera.near_plane
    near_clip_r = camera.pos + camera.near_dir + camera.near_plane
    final_pts = []

    #add rect corner points that are within camera field of vision bounds
    for pt in rect:
        if point_in_triangle(pt, near_clip_l, near_clip_r, camera.pos):  #exclude points that are behind the near clip plane
            continue
        vec = pt - camera.pos
        unit = vec / norm2(vec)
        if (np.dot(cam_l_ray, unit) > (1 - 1e-4) or
            np.dot(cam_r_ray, unit) > (1 - 1e-4)):
            final_pts.append(pt)
        elif (np.sign(np.cross(cam_l_ray, vec)) == np.sign(np.cross(vec, cam_r_ray)) and
            np.dot(vec, camera.near_dir) >= 0):
            final_pts.append(pt)

    #rect is partially in view frustrum, find intersection points with camera bounds
    if len(final_pts) < 4:
        for edge in edges:
            pt = intersect_ray_segment(near_clip_l, cam_l_ray, edge[0], edge[1])
            if pt is not None:
                final_pts.append(pt)
            pt = intersect_ray_segment(near_clip_r, cam_r_ray, edge[0], edge[1])
            if pt is not None:
                final_pts.append(pt)
            pt = intersect_segments(edge[0], edge[1], near_clip_l, near_clip_r)
            if pt is not None and point_in_rect(pt, tl, br):
                final_pts.append(pt)

    #add endpoints of near clip plane if inside the rect
    if point_in_rect(near_clip_l, tl, br):
        final_pts.append(near_clip_l)
    if point_in_rect(near_clip_r, tl, br):
        final_pts.append(near_clip_r)

    #sort points
    final_pts = sort_cw(final_pts)

    return final_pts

def clip_wall(segment, camera):
    cam_l_ray = camera.rays[0]
    cam_r_ray = camera.rays[-1]
    final_pts = []

    #add points where wall intersects field of vision edges
    for ray in (cam_l_ray, cam_r_ray):
        pt = intersect_ray_segment(camera.pos, ray, segment[0], segment[1])
        if pt is not None:
            final_pts.append(pt)

    #add wall endpoints if they're within field of vision bounds
    for pt in segment:
        vec = pt - camera.pos
        if (np.sign(np.cross(cam_l_ray, vec)) == np.sign(np.cross(vec, cam_r_ray)) and  #point is within camera angle bounds
            np.dot(vec, camera.near_dir) >= 0):  #point is within 180 degrees of camera direction
            final_pts.append(pt)

    #final wall bounds will be the left-most and right-most points
    final_pts = np.array(final_pts)
    clip_left = np.array([np.min(final_pts[:,0]), np.min(final_pts[:,1])])
    clip_right = np.array([np.max(final_pts[:,0]), np.max(final_pts[:,1])])

    return [clip_left, clip_right]

def get_clipped_tile_points(tilemap, camera):
    floor_pts = []
    wall_pts = []
    used_tiles = set()
    tile_size = tilemap.size
    max_z = 64
    min_z = -64

    for ray in camera.rays:
        stop_up = False
        stop_down = False
        prev_min_z = 0
        prev_max_z = 0
        prev_floor_z = 0
        for collision, side in dda(camera.pos, ray, float(tile_size)):
            collision_int = [int(collision[X]), int(collision[Y])]

            if (collision_int[X] >= tilemap.width * tilemap.size or collision_int[X] < 0 or
                collision_int[Y] >= tilemap.height * tilemap.size or collision_int[Y] < 0):
                break

            if stop_up and stop_down:
                break

            walls, wall_coords = tilemap.get_walls_px(collision_int[X], collision_int[Y])
            render_walls = map(lambda w: ((w.z + w.height > prev_max_z and not stop_up) or (w.z + w.height > prev_floor_z)) and w.height > 0, walls)

            if len(walls) == 0:
                wall_min_z = prev_min_z
                wall_max_z = prev_max_z
            else:
                wall_min_z = min(map(lambda x: x.z, walls))
                wall_max_z = max(map(lambda x: x.z + x.height, walls))

            if wall_max_z >= max_z:
                stop_up = True

            if wall_min_z <= min_z:
                stop_down = True

            prev_max_z = wall_max_z
            prev_min_z = wall_min_z

            floor = tilemap.get_floor_px(collision_int[X], collision_int[Y])
            render_floor = True if camera.z > wall_max_z and wall_max_z < max_z and floor.texture is not None else False
            prev_floor_z = floor.z + floor.height

            #skip processing if collision point is almost a tile corner
            if (abs(collision_int[X] % tile_size - tile_size) <= 1 or abs(collision_int[X] % tile_size) <= 1) and (abs(collision_int[Y] % tile_size - tile_size) <= 1 or abs(collision_int[Y] % tile_size) <= 1):
                continue

            #floor
            if render_floor:
                tile_coords = [collision_int[X] / tilemap.size, collision_int[Y] / tile_size]
                key = (tile_coords[X], tile_coords[Y])
                if key not in used_tiles:
                    used_tiles.add(key)
                    #[tl, tr, br, bl]
                    l = round_down(collision[X], tile_size)
                    r = round_up(collision[X], tile_size)
                    t = round_down(collision[Y], tile_size)
                    b = round_up(collision[Y], tile_size)
                    rect = np.array([
                        [l, t], [r, t], [r, b], [l, b]
                        ])
                    clip_pts = clip_floor(rect, camera)
                    if len(clip_pts) > 0:
                        floor_pts.append(Clip(TYPE_FLOOR, clip_pts, [rect[0], rect[2]], floor))

            #wall
            if collision[X] != camera.pos[X] and collision[Y] != camera.pos[Y]:
                for wall, wall_coord, render_wall in zip(walls, wall_coords, render_walls):
                    key = (wall_coord[X], wall_coord[Y], side)
                    if key not in used_tiles and render_wall:
                        used_tiles.add(key)
                        if side == 0:  #horizontal
                            y0 = y1 = round_nearest(collision[Y], tile_size)
                            x0 = round_down(collision[X], tile_size)
                            x1 = round_up(collision[X], tile_size)
                            # if collision[Y] % tile_size == 0:  #bottom wall, flip x's
                            #   x0, x1 = x1, x0
                        else:  #vertical
                            x0 = x1 = round_nearest(collision[X], tile_size)
                            y1 = round_down(collision[Y], tile_size)
                            y0 = round_up(collision[Y], tile_size)
                            # if collision[X] % tile_size == 0:  #right wall, flip y's
                            #   y0, y1 = y1, y0
                        segment = np.array([[x0, y0], [x1, y1]])
                        clip_pts = clip_wall(segment, camera)
                        if len(clip_pts) > 0:
                            wall_pts.append(Clip(TYPE_WALL, clip_pts, [segment[0], segment[1]], wall))

    return floor_pts, wall_pts

def get_tri_quads(clips, camera):
    view_center = np.array([camera.proj_width / 2, camera.proj_height / 2])
    all_tri_quads = []
    all_view_quads = []
    final_quads = []

    for clip in clips:
        surface_type = clip.type
        data = clip.data

        #add z coordinate to clip points
        if surface_type == TYPE_FLOOR:
            clip_points_3D = np.empty((len(clip.points), 3))
            clip_points_3D[:,0:2] = clip.points
            clip_points_3D[:,2] = data.z + data.height
        elif surface_type == TYPE_WALL:
            clip_points_3D = np.empty((4, 3))
            clip_points_3D[:,0:2] = (clip.points[0], clip.points[1], clip.points[1], clip.points[0])
            clip_points_3D[:,2] = (data.z + data.height, data.z + data.height, data.z, data.z)

        #triangulate the 3D clip points
        tris = triangulate(clip_points_3D)
        if len(tris) == 0:
            continue

        #append the triangle midpoints as the 4th point to form a tri-quad
        tri_quads = np.empty((len(tris), 4, 3))
        tri_quads[:,0:3] = tris
        tri_quads[:,3] = np.average(tri_quads[:,0:3], axis=1)

        #rotate tri-quad onto the viewport (XY) plane, at original size
        if surface_type is TYPE_FLOOR:
            view_tri_quads = tri_quads[:,:,0:2]
        elif surface_type is TYPE_WALL:
            rot_mat = MATRIX_ROT_YZ_XY if tri_quads[0][0][X] == tri_quads[0][1][X] else MATRIX_ROT_XZ_XY
            view_tri_quads = np.dot(tri_quads, rot_mat.T)[:,:,0:2]

        #normalized distance of tri-quad points from the tile's origin point
        if surface_type == TYPE_FLOOR:
            tile_origin = clip.bounds[0]
            tile_size = clip.bounds[1] - clip.bounds[0]
        elif surface_type == TYPE_WALL:
            tile_origin = clip.bounds[0]
            tile_size = np.array([norm2(clip.bounds[1] - clip.bounds[0]), data.height])
        offsets = (view_tri_quads - tile_origin) / tile_size

        all_tri_quads += list(tri_quads)
        all_view_quads += list(view_tri_quads)
        final_quads += [[None, None, None, offset, data.texture, surface_type] for offset in offsets]

    if len(all_tri_quads) > 0:
        #translate view tri-quads to the center of the viewport
        view_quads_arr = np.array(all_view_quads)
        d_mids = view_center - view_quads_arr[:,3]
        orig_pts = np.empty((len(view_quads_arr), 4, 3))
        orig_pts[:,:,0:2] = view_quads_arr + d_mids[:,None]
        orig_pts[:,:,2] = 0
        normalize_geoms(orig_pts, camera)

        #project tri-quads onto the viewport
        tri_quads_arr = np.array(all_tri_quads)
        proj_pts = project_points(tri_quads_arr.reshape(len(tri_quads_arr) * 4, 3), camera)
        proj_pts = proj_pts.reshape((len(tri_quads_arr), 4, 3))
        normalize_geoms(proj_pts, camera)
        proj_mid_pts = np.average(proj_pts[:,0:3], axis=1)
        proj_pts[:,:,0] -= proj_mid_pts[:,0,None]
        proj_pts[:,:,1] -= proj_mid_pts[:,1,None]

        for i in xrange(len(final_quads)):
            final_quads[i][0] = orig_pts[i]
            final_quads[i][1] = proj_pts[i]
            final_quads[i][2] = proj_mid_pts[i]

    return final_quads
