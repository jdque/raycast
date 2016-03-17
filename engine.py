import math
import numpy as np

from geometry import *

X = 0
Y = 1
Z = 2
W = 2
H = 3

class TilePalette:
	def __init__(self):
		self.palette = {}

	def add(self, index, kind, floor_height, floor_z, floor_tex, wall_tex):
		self.palette[str(index)] = {
			"kind": kind,
			"floor_height": floor_height,
			"floor_z": floor_z,
			"floor_tex": floor_tex,
			"wall_tex": wall_tex,
			}

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
		for y in range(0, self.height):
			for x in range(0, self.width):
				self.tiles[y][x] = palette.get(tiles[y][x])

	def get_tile(self, x, y):
		return self.tiles[y][x]

	def get_tile_px(self, x, y):
		tx = int(x / self.size)
		ty = int(y / self.size)
		if tx < 0 or tx >= self.width or ty < 0 or ty >= self.height:
			return None
		return self.tiles[ty][tx]

class Camera:
	def __init__(self):
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
		self.dir = np.dot(rot, self.dir)
		self.plane = np.dot(rot, self.plane)
		self.near_dir = np.dot(rot, self.near_dir)
		self.near_plane = np.dot(rot, self.near_plane)

		for i in range(0, len(self.rays)):
			self.rays[i] = np.dot(rot, self.rays[i])

	def move_forward(self, distance):
		step = self.dir / np.linalg.norm(self.dir) * distance
		self.move_by(step[0], step[1], 0)

	def tilt_by(self, distance):
		self.horizon_y += distance

	def set_fov(self, angle, near, far, proj_width, proj_height):
		self.angle = float(angle)
		self.near = float(near)
		self.far = float(far)
		self.proj_width = float(proj_width)
		self.proj_height = float(proj_height)
		self.horizon_y = float(proj_height) / 2

		self.dir = np.array([(self.proj_width / 2) / np.tan(np.deg2rad(self.angle / 2)), 0])
		self.plane = np.array([0, self.proj_width / 2])
		self.near_dir = np.array([self.near, 0])
		self.near_plane = np.array([0, self.near * np.tan(np.deg2rad(self.angle / 2))])
		self.rays = self.generate_rays()

	def generate_rays(self):
		rays = []
		unit_plane = self.plane / math.sqrt(self.plane[0]**2 + self.plane[1]**2)
		for i in range(int(-self.proj_width / 2), int(self.proj_width / 2)):
			plane_pt = self.dir + (unit_plane * i)
			unit_ray = plane_pt / math.sqrt(plane_pt[0]**2 + plane_pt[1]**2)
			rays.append(unit_ray)
		return np.array(rays)

def dda(origin, dir, step, tilemap):
	m = dir[1] / dir[0]

	if dir[0] >= 0:
		if dir[1] < 0: #1
			dx = 1
			dy = -1
		else: #4
			dx = 1
			dy = 1
	else:
		if dir[1] < 0: #2
			dx = -1
			dy = -1
		else: #3
			dx = -1
			dy = 1

	delt_x = math.sqrt(step**2 + (m * step)**2)
	delt_y = math.sqrt(step**2 + (1/m * step)**2)

	if dx > 0:
		side_x = abs(step - origin[0] % step) * delt_x / step + 1
	else:
		side_x = abs(origin[0] % step) * delt_x / step + 1

	if dy > 0:
		side_y = abs(step - origin[1] % step) * delt_y / step + 1
	else:
		side_y = abs(origin[1] % step) * delt_y / step + 1

	x = side_x
	y = side_y

	yield origin, None

	while True:
		if x < y:
			hit = origin + dir * abs(x)
			side = 1
			x += delt_x
		else:
			hit = origin + dir * abs(y)
			side = 0
			y += delt_y
		yield hit, side

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
		scaled_rej_len = np.linalg.norm(camera.near_dir) / proj_len * rej_len
		scaled_rej = rej / rej_len * scaled_rej_len
		x_sign = np.sign(np.dot((camera.near_plane + scaled_rej), camera.near_plane))
		x = x_sign * np.linalg.norm(camera.near_plane + scaled_rej)

	#y
	vector = np.array([proj_len, camera.z - z])
	ndir = np.array([np.linalg.norm(camera.near_dir), 0])
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

	return [x * (np.linalg.norm(camera.plane) / np.linalg.norm(camera.near_plane)),
			y * (np.linalg.norm(camera.plane) / np.linalg.norm(camera.near_plane)) - camera.horizon_y,
			z]

def normalize_projection_points(pts, camera):
	pts[:,0:2] /= [camera.proj_width, camera.proj_height]
	pts[:,0:2] -= 0.5
	pts[:,0:2] *= 2

def clip_floor(ul, ur, br, bl, camera):
	edges = [
		[ul, bl],
		[ur, br],
		[ul, ur],
		[bl, br]
		]
	cam_l_ray = camera.rays[0]
	cam_r_ray = camera.rays[-1]

	final_pts = []

	#filter points that are outside the view frustrum
	for pt in [ul, ur, bl, br]:
		if not point_in_rect(pt, ul, br) or point_in_triangle(pt, camera.pos + (camera.near_dir * 0.99) - (camera.near_plane * 1.01), camera.pos + (camera.near_dir * 0.99) + (camera.near_plane * 0.99), camera.pos):
			continue
		unit = (pt - camera.pos) / np.linalg.norm(pt - camera.pos)
		if np.dot(cam_l_ray / np.linalg.norm(cam_l_ray), unit) > 0.9999:
			final_pts.append(pt)
		elif np.dot(cam_r_ray / np.linalg.norm(cam_r_ray), unit) > 0.9999:
			final_pts.append(pt)
		elif np.sign(np.cross(cam_l_ray, pt - camera.pos)) == np.sign(np.cross(pt - camera.pos, cam_r_ray)) and np.dot(pt - camera.pos, camera.dir) >= 0:
			final_pts.append(pt)

	#add points where tile edges intersect field of vision bounds
	clip_rays = [
		[camera.pos + camera.near_dir - camera.near_plane, cam_l_ray / np.linalg.norm(cam_l_ray)],
		[camera.pos + camera.near_dir + camera.near_plane, cam_r_ray / np.linalg.norm(cam_r_ray)]
		]
	for clip_ray in clip_rays:
		for edge in edges:
			pt = intersect_ray_segment(clip_ray[0], clip_ray[1], edge[0], edge[1])
			if pt is not None:
				final_pts.append(pt)

	#add points where tile edges intersect near clip segment
	clip_segs = [
		[camera.pos + camera.near_dir - camera.near_plane, camera.pos + camera.near_dir + camera.near_plane]
		]
	for clip_seg in clip_segs:
		for edge in edges:
			pt = intersect_segments(edge[0], edge[1], clip_seg[0], clip_seg[1])
			if pt is not None and point_in_rect(pt, ul, br):
				final_pts.append(pt)

	#add left and right endpoints of near clip segment if they are inside the tile
	near_clip_l = camera.pos + camera.near_dir - camera.near_plane
	if point_in_rect(near_clip_l, ul, br):
		final_pts.append(near_clip_l)
	near_clip_r = camera.pos + camera.near_dir + camera.near_plane
	if point_in_rect(near_clip_r, ul, br):
		final_pts.append(near_clip_r)

	#sort points
	final_pts = sort_cw(final_pts)

	return final_pts

def clip_wall(left, right, camera):
	cam_l_ray = camera.rays[0]
	cam_r_ray = camera.rays[-1]

	final_pts = []

	#add points where wall intersects field of vision bounds
	clip_rays = [
		[camera.pos, cam_l_ray / np.linalg.norm(cam_l_ray)],
		[camera.pos, cam_r_ray / np.linalg.norm(cam_r_ray)]
		]
	for clip_ray in clip_rays:
		pt = intersect_ray_segment(clip_ray[0], clip_ray[1], left, right)
		if pt is not None:
			final_pts.append(pt)

	#add wall endpoints if they're within field of vision bounds
	if np.sign(np.cross(cam_l_ray, left - camera.pos)) == np.sign(np.cross(left - camera.pos, cam_r_ray)) and np.dot(left - camera.pos, camera.dir) >= 0:
		final_pts.append(left)
	if np.sign(np.cross(cam_l_ray, right - camera.pos)) == np.sign(np.cross(right - camera.pos, cam_r_ray)) and np.dot(right - camera.pos, camera.dir) >= 0:
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
	max_z = 64

	for ray in camera.rays:
		stop = False
		prev_z = None
		occluded = False
		for collision, side in dda(camera.pos, ray, 64, tilemap):
			tile = tilemap.get_tile_px(collision[X], collision[Y])

			if tile is None or stop:
				break

			tile_coords = [int(collision[X]) / 64, int(collision[Y]) / 64]
			tile_height = tile["floor_height"]
			tile_z = tile["floor_z"]

			render_wall = True if tile_z + tile_height > prev_z and prev_z is not None else False
			render_floor = True if camera.z - tile_z + tile_height > 0 and not occluded and tile_z + tile_height < max_z else False

			if tile_z + tile_height >= max_z:
				stop = True

			if tile_z + tile_height - camera.z >= 0:
				occluded = True

			prev_z = tile_z + tile_height

			if (int(collision[X]) + 1) % 64 <= 1.0 and (int(collision[Y]) + 1) % 64 <= 1.0:
				continue

			#floor
			if render_floor:
				key = (tile_coords[X], tile_coords[Y])
				if key not in used_tiles:
					used_tiles.add(key)
					ul = np.array([tile_coords[X] * 64, tile_coords[Y] * 64])
					ur = np.array([tile_coords[X] * 64 + 64, tile_coords[Y] * 64])
					br = np.array([tile_coords[X] * 64 + 64, tile_coords[Y] * 64 + 64])
					bl = np.array([tile_coords[X] * 64, tile_coords[Y] * 64 + 64])
					clip_pts = clip_floor(ul, ur, br, bl, camera)
					floor_pts.append((clip_pts, [ul, br], tile, 0))

			#wall
			if render_wall:
				key = (tile_coords[X], tile_coords[Y], side)
				if key not in used_tiles and render_wall:
					used_tiles.add(key)
					if side == 0:
						y = int(collision[Y] + 32 - (collision[Y] + 32) % 64)
						if int(collision[Y]) % 64 <= 1.0:
							left = np.array([tile_coords[X] * 64 + 64, y])
							right = np.array([tile_coords[X] * 64, y])
						else:
							left = np.array([tile_coords[X] * 64, y])
							right = np.array([tile_coords[X] * 64 + 64, y])
					else:
						x = int(collision[X] + 32 - (collision[X] + 32) % 64)
						if int(collision[X]) % 64 <= 1.0:
							left = np.array([x, tile_coords[Y] * 64])
							right = np.array([x, tile_coords[Y] * 64 + 64])
						else:
							left = np.array([x, tile_coords[Y] * 64 + 64])
							right = np.array([x, tile_coords[Y] * 64])
					clip_pts = clip_wall(left, right, camera)
					wall_pts.append((clip_pts, [left, right], tile, 1))

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
					trans_pt = project_point(pt,  tile["floor_z"] + tile["floor_height"], camera)
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
				top_pt = project_point(pt, tile["floor_z"] + tile["floor_height"], camera)
				trans_pts.append(top_pt)
				#bottom
				bottom_pt = project_point(pt, tile["floor_z"], camera)
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
