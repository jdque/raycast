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

Tile = namedtuple('Tile', ['kind', 'floor_height', 'floor_z', 'floor_tex', 'wall_tex'])

class TilePalette:
	def __init__(self):
		self.palette = {}

	def add(self, index, tile):
		self.palette[str(index)] = tile

	def get(self, index):
		return self.palette[str(index)]


class TileMap:
	def __init__(self, width, height, size):
		self.width = width
		self.height = height
		self.size = size
		self.tiles = np.zeros([self.width, self.height], dtype=object)

	def set_tiles_from_palette(self, palette, tiles):
		if len(tiles[0]) != self.width or len(tiles) != self.height:
			return
		for y in xrange(0, self.height):
			for x in xrange(0, self.width):
				self.tiles[y][x] = palette.get(tiles[y][x])

	def get_tile(self, x, y):
		return self.tiles[y][x]

	def get_tile_px(self, x, y):
		tx = x / self.size
		ty = y / self.size
		if tx < 0 or tx >= self.width or ty < 0 or ty >= self.height:
			return None
		return self.tiles[ty,tx]

class Camera:
	def __init__(self):
		self.pos = None
		self.z = 0.
		self.angle = 0.
		self.near = 0.
		self.far = 0.
		self.proj_width = 0.
		self.proj_height = 0.
		self.aspect = 0.
		self.horizon_y = 0.
		self.near_dir = None
		self.near_plane = None
		self.rays = None

		self.pos = np.array([0., 0.])
		self.z = 0.
		self.set_fov(90, 0, 100, 100, 100)

	def move_to(self, x, y, z):
		self.pos[0] = x
		self.pos[1] = y
		self.z = z

	def rotate_to(self, angle):
		rad = np.deg2rad(angle)
		rot = np.array([
			[np.cos(rad), -np.sin(rad)],
			[np.sin(rad), np.cos(rad)]
		])
		self.dir = np.dot(rot, self.dir)
		self.plane = np.dot(rot, self.plane)
		self.near_dir = np.dot(rot, self.near_dir)
		self.near_plane = np.dot(rot, self.near_plane)

	def move_by(self, dx, dy, dz):
		self.pos[0] += dx
		self.pos[1] += dy
		self.z += dz

	def rotate_by(self, da):
		rad = np.deg2rad(da)
		rot = np.array([
			[np.cos(rad), -np.sin(rad)],
			[np.sin(rad), np.cos(rad)]
		])
		self.near_dir = np.dot(rot, self.near_dir)
		self.near_plane = np.dot(rot, self.near_plane)

		for i in xrange(0, len(self.rays)):
			self.rays[i] = np.dot(rot, self.rays[i])

	def move_forward(self, distance):
		step = self.near_dir / self.near * distance
		self.move_by(step[0], step[1], 0)

	def tilt_by(self, distance):
		self.horizon_y += distance

	def set_fov(self, angle, near, far, proj_width, proj_height):
		self.angle = float(angle)
		self.near = float(near)
		self.far = float(far)
		self.proj_width = float(proj_width)
		self.proj_height = float(proj_height)
		self.aspect = float(self.proj_width / self.proj_height)
		self.horizon_y = float(proj_height) / 2

		self.near_dir = np.array([self.near, 0])
		self.near_plane = np.array([0, self.near * np.tan(np.deg2rad(self.angle / 2))])
		self.rays = self.generate_rays()

	def generate_rays(self):
		rays = []
		dir_vec = np.array([(self.proj_width / 2) / np.tan(np.deg2rad(self.angle / 2)), 0])
		unit_plane = np.array([0, 1])
		for i in xrange(int(-self.proj_width / 2), int(self.proj_width / 2)):
			plane_pt = dir_vec + (unit_plane * i)
			unit_ray = plane_pt / math.sqrt(plane_pt[0]**2 + plane_pt[1]**2)
			rays.append(unit_ray)
		return np.array(rays)

def project_point(pt, z, camera):
	#x
	vector = pt - camera.pos
	if abs(1 - np.dot(vector, camera.near_dir) / (np.linalg.norm(vector) * camera.near)) < 1e-10:
		proj_len = np.linalg.norm(vector)
		x = np.linalg.norm(camera.near_plane)
	else:
		proj = camera.near_dir * (np.dot(vector, camera.near_dir) / np.dot(camera.near_dir, camera.near_dir))
		rej = vector - proj
		proj_len = np.linalg.norm(proj)
		rej_len = np.linalg.norm(rej)
		scaled_rej_len = camera.near / proj_len * rej_len
		scaled_rej = rej / rej_len * scaled_rej_len
		x_sign = np.sign(np.dot((camera.near_plane + scaled_rej), camera.near_plane))
		x = x_sign * np.linalg.norm(camera.near_plane + scaled_rej)

	#y
	vector = np.array([proj_len, camera.z - z])
	ndir = np.array([camera.near, 0])
	nplane = np.array([0, np.linalg.norm(camera.near_plane)])
	if abs(1 - np.dot(vector, ndir) / (np.linalg.norm(vector) * camera.near)) < 1e-10:
		y = np.linalg.norm(nplane)
	else:
		proj = ndir * (np.dot(vector, ndir) / np.dot(ndir, ndir))
		rej = vector - proj
		proj_len = np.linalg.norm(proj)
		rej_len = np.linalg.norm(rej)
		scaled_rej_len = np.linalg.norm(ndir) / proj_len * rej_len
		scaled_rej = rej / rej_len * scaled_rej_len
		y_sign = np.sign(np.dot((nplane + scaled_rej), nplane))
		y = y_sign * np.linalg.norm(nplane + scaled_rej)

	#z
	vector = pt - camera.pos
	proj = camera.near_dir * (np.dot(vector, camera.near_dir) / np.dot(camera.near_dir, camera.near_dir))
	z = np.linalg.norm(proj) / (camera.far - camera.near)

	return [x * (camera.proj_width / camera.near),
			y * (camera.proj_height / camera.near) * camera.aspect - camera.horizon_y,
			z]

def normalize_projection_points(pts, camera):
	pts[:,0:2] /= [camera.proj_width, camera.proj_height]
	pts[:,0:2] -= 0.5
	pts[:,0:2] *= 2

def round_down(number, multiple = 10):
	return number - (number % multiple)

def round_up(number, multiple = 10):
	return number - (number % multiple) + multiple

def round(number, multiple = 10):
	rem = number % multiple
	return (number - rem + multiple) if rem >= multiple / 2 else number - rem

def clip_floor(corners, camera):
	tl = corners[0]
	tr = corners[1]
	br = corners[2]
	bl = corners[3]
	edges = [
		[tl, bl],
		[tr, br],
		[tl, tr],
		[bl, br]
		]
	cam_l_ray = camera.rays[0]
	cam_r_ray = camera.rays[-1]

	near_clip_l = camera.pos + camera.near_dir - camera.near_plane
	near_clip_r = camera.pos + camera.near_dir + camera.near_plane

	final_pts = []

	#filter points that are outside the view frustrum
	for pt in corners:
		if point_in_triangle(pt, near_clip_l, near_clip_r, camera.pos):
			continue
		vector = pt - camera.pos
		unit = vector / np.linalg.norm(vector)
		if np.dot(cam_l_ray, unit) > 0.9999:
			final_pts.append(pt)
		elif np.dot(cam_r_ray, unit) > 0.9999:
			final_pts.append(pt)
		elif np.sign(np.cross(cam_l_ray, vector)) == np.sign(np.cross(vector, cam_r_ray)) and np.dot(vector, camera.near_dir) >= 0:
			final_pts.append(pt)

	#tile is partially in view frustrum, find intersection points with camera bounds
	if len(final_pts) < 4:
		for edge in edges:
			pt = intersect_ray_segment(near_clip_l, cam_l_ray, edge[0], edge[1])
			if pt is not None and point_in_rect(pt, tl, br):
				final_pts.append(pt)
			pt = intersect_ray_segment(near_clip_r, cam_r_ray, edge[0], edge[1])
			if pt is not None and point_in_rect(pt, tl, br):
				final_pts.append(pt)
			pt = intersect_segments(edge[0], edge[1],near_clip_l, near_clip_r)
			if pt is not None and point_in_rect(pt, tl, br):
				final_pts.append(pt)

	#add endpoints of near clip plane if inside the tile
	if point_in_rect(near_clip_l, tl, br):
		final_pts.append(near_clip_l)
	if point_in_rect(near_clip_r, tl, br):
		final_pts.append(near_clip_r)

	#sort points
	final_pts = sort_cw(final_pts)

	return final_pts

def clip_wall(edges, camera):
	left = edges[0]
	right = edges[1]
	cam_l_ray = camera.rays[0]
	cam_r_ray = camera.rays[-1]

	final_pts = []

	#add points where wall intersects field of vision bounds
	pt = intersect_ray_segment(camera.pos, cam_l_ray, left, right)
	if pt is not None:
		final_pts.append(pt)
	pt = intersect_ray_segment(camera.pos, cam_r_ray, left, right)
	if pt is not None:
		final_pts.append(pt)

	#add wall endpoints if they're within field of vision bounds
	l_vector = left - camera.pos
	if np.sign(np.cross(cam_l_ray, l_vector)) == np.sign(np.cross(l_vector, cam_r_ray)) and np.dot(l_vector, camera.near_dir) >= 0:
		final_pts.append(left)
	r_vector = right - camera.pos
	if np.sign(np.cross(cam_l_ray, r_vector)) == np.sign(np.cross(r_vector, cam_r_ray)) and np.dot(r_vector, camera.near_dir) >= 0:
		final_pts.append(right)

	#final wall bounds will be the left-most and right-most points
	final_pts = np.array(final_pts)
	clip_left = np.array([np.min(final_pts[:,0]), np.min(final_pts[:,1])])
	clip_right = np.array([np.max(final_pts[:,0]), np.max(final_pts[:,1])])

	return [clip_left, clip_right]

def get_clipped_tile_points(tilemap, camera):
	floor_pts = []
	wall_pts = []
	used_tiles = set()
	tile_size = float(tilemap.size)
	max_z = 64

	for ray in camera.rays:
		stop = False
		prev_z = None
		occluded = False
		for collision, side in dda(camera.pos, ray, tile_size):
			collision_int = [int(collision[X]), int(collision[Y])]
			tile = tilemap.get_tile_px(collision_int[X], collision_int[Y])

			if tile is None or stop:
				break

			tile_coords = [int(collision_int[X] / tile_size), int(collision_int[Y] / tile_size)]
			wall_z = tile.floor_z + tile.floor_height

			render_wall = True if wall_z > prev_z and prev_z is not None else False
			render_floor = True if camera.z - wall_z > 0 and not occluded and wall_z < max_z else False

			if wall_z >= max_z:
				stop = True

			if wall_z - camera.z >= 0:
				occluded = True

			prev_z = wall_z

			#skip processing if collision point is almost a tile corner
			if (collision_int[X] + 1) % tile_size <= 1.0 and (collision_int[Y] + 1) % tile_size <= 1.0:
				continue

			#floor
			if render_floor:
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
					floor_pts.append((clip_pts, [rect[0], rect[2]], tile, 0))

			#wall
			if render_wall:
				key = (tile_coords[X], tile_coords[Y], side)
				if key not in used_tiles and render_wall:
					used_tiles.add(key)
					if side == 0:  #horizontal
						y0 = y1 = round(collision[Y], tile_size)
						x0 = round_down(collision[X], tile_size)
						x1 = round_up(collision[X], tile_size)
						if collision[Y] % tile_size == 0:  #bottom wall, flip x's
							x0, x1 = x1, x0
					else:  #vertical
						x0 = x1 = round(collision[X], tile_size)
						y0 = round_down(collision[Y], tile_size)
						y1 = round_up(collision[Y], tile_size)
						if collision[X] % tile_size == 0:  #right wall, flip y's
							y0, y1 = y1, y0
					segment = np.array([[x0, y0], [x1, y1]])
					clip_pts = clip_wall(segment, camera)
					wall_pts.append((clip_pts, [segment[0], segment[1]], tile, 1))

	return floor_pts, wall_pts

def get_tri_quads(tile_pts, camera):
	final_quads = []
	for tile_pt in tile_pts:
		tile = tile_pt[2]
		surface_type = tile_pt[3]
		if surface_type == 0: #floor
			tile_x = tile_pt[1][0][0]
			tile_y = tile_pt[1][0][1]
			recp_tile_w = 1. / (tile_pt[1][1][0] - tile_pt[1][0][0])
			recp_tile_h = 1. / (tile_pt[1][1][1] - tile_pt[1][0][1])

			tri_quads = []
			tris = triangulate(tile_pt[0])
			for tri in tris:
				mid_pt = np.average(tri, axis=0)
				tri_quads.append([tri[0], tri[1], tri[2], mid_pt])

			for tri_quad in tri_quads:
				og_pts = []
				d_mid = np.array([camera.proj_width / 2 - tri_quad[3][X], camera.proj_height / 2 - tri_quad[3][Y]])
				for pt in tri_quad:
					d = pt + d_mid
					og_pts.append([d[X], d[Y], 0.0])
				og_pts = np.array(og_pts)
				normalize_projection_points(og_pts, camera)

				trans_pts = []
				for pt in tri_quad:
					trans_pt = project_point(pt,  tile.floor_z + tile.floor_height, camera)
					trans_pts.append(trans_pt)
				trans_pts = np.array(trans_pts)
				normalize_projection_points(trans_pts, camera)

				trans_mid_pt = np.average(trans_pts[0:3], axis=0)
				trans_pts[:,0:2] -= trans_mid_pt[0:2]

				offsets = np.array([
					[(tri_quad[0][X] - tile_x) * recp_tile_w, (tri_quad[0][Y] - tile_y) * recp_tile_h],
					[(tri_quad[1][X] - tile_x) * recp_tile_w, (tri_quad[1][Y] - tile_y) * recp_tile_h],
					[(tri_quad[2][X] - tile_x) * recp_tile_w, (tri_quad[2][Y] - tile_y) * recp_tile_h],
					[(tri_quad[3][X] - tile_x) * recp_tile_w, (tri_quad[3][Y] - tile_y) * recp_tile_h],
					])

				final_quads.append((og_pts, trans_pts, trans_mid_pt, offsets, tile, surface_type))
		elif surface_type == 1: #wall
			#temp - append tri midpoints
			clip_w = abs(tile_pt[0][1] - tile_pt[0][0])
			tile_pt[0].append(tile_pt[0][0] + clip_w * 1/3.)
			tile_pt[0].append(tile_pt[0][1] - clip_w * 1/3.)

			trans_pts = [] #[ul, bl, ur, br]
			for pt in tile_pt[0]:
				#top
				top_pt = project_point(pt, tile.floor_z + tile.floor_height, camera)
				trans_pts.append(top_pt)
				#bottom
				bottom_pt = project_point(pt, tile.floor_z, camera)
				trans_pts.append(bottom_pt)
			trans_pts = np.array(trans_pts)
			normalize_projection_points(trans_pts, camera)

			#get offsets relative to tile width
			recp_tile_w = 1. / np.linalg.norm(tile_pt[1][1] - tile_pt[1][0])
			offset_l = np.linalg.norm(tile_pt[0][0] - tile_pt[1][0]) * recp_tile_w
			offset_r = np.linalg.norm(tile_pt[0][1] - tile_pt[1][0]) * recp_tile_w

			"""
			-------
			|\    |
			| \ 2 |
			|  \  |
			| 1 \ |
			|    \|
			-------
			"""
			w = np.linalg.norm(tile_pt[0][1] - tile_pt[0][0])
			o_l = camera.proj_width / 2. - w / 2.
			o_r = camera.proj_width / 2. + w / 2.
			o_t = camera.proj_height / 2. - 32
			o_b = camera.proj_height / 2. + 32
			o_ul = [o_l, o_t, 0.0]
			o_ur = [o_r, o_t, 0.0]
			o_br = [o_r, o_b, 0.0]
			o_bl = [o_l, o_b, 0.0]

			l_third = abs(trans_pts[1] - trans_pts[0]) * 1/3.
			r_third = abs(trans_pts[3] - trans_pts[2]) * 1/3.

			#triquad 1
			og_pts = np.array([
				o_ul,
				o_br,
				o_bl,
				np.average([o_ul, o_br, o_bl], axis=0)
				])
			normalize_projection_points(og_pts, camera)
			proj_mid = intersect_segments(
				trans_pts[1][0:2] - l_third[0:2], trans_pts[3][0:2] - r_third[0:2], trans_pts[4][0:2], trans_pts[5][0:2])
			if proj_mid is None:
				continue
			tr_pts = np.array([
				trans_pts[0],
				trans_pts[3],
				trans_pts[1],
				[proj_mid[X], proj_mid[Y], 0.0]
				])
			mid = np.average(tr_pts[0:3], axis=0)
			tr_pts[:,0:2] -= mid[0:2]
			offs = np.array([
				[offset_l, 1.0],
				[offset_r, 0.0],
				[offset_l, 0.0],
				[(offset_l + offset_r + offset_l) / 3., 1/3.],
				])
			final_quads.append((og_pts, tr_pts, mid, offs, tile, surface_type))

			#triquad 2
			og_pts = np.array([
				o_ul,
				o_ur,
				o_br,
				np.average([o_ul, o_ur, o_br], axis=0)
				])
			normalize_projection_points(og_pts, camera)
			proj_mid = intersect_segments(
				trans_pts[0][0:2] + l_third[0:2], trans_pts[2][0:2] + r_third[0:2], trans_pts[6][0:2], trans_pts[7][0:2])
			tr_pts = np.array([
				trans_pts[0],
				trans_pts[2],
				trans_pts[3],
				[proj_mid[X], proj_mid[Y], 0.0]
				])
			mid = np.average(tr_pts[0:3], axis=0)
			tr_pts[:,0:2] -= mid[0:2]
			offs = np.array([
				[offset_l, 1.0],
				[offset_r, 1.0],
				[offset_r, 0.0],
				[(offset_l + offset_r + offset_r) / 3., 2/3.],
				])
			final_quads.append((og_pts, tr_pts, mid, offs, tile, surface_type))

	return final_quads
