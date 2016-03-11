import math
import numpy as np

from geometry import *

X = 0
Y = 1
W = 2
H = 3

class TilePalette:
	def __init__(self):
		self.palette = {}

	def add(self, index, kind, floor_height, floor_tex, wall_tex):
		self.palette[str(index)] = {
			"kind": kind,
			"floor_height": floor_height,
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
		if tx >= self.width or ty >= self.height:
			return None
		return self.tiles[ty][tx]

class Camera:
	def __init__(self):
		self.pos = np.array([0., 0.])
		self.dir = np.array([100, 0])
		self.plane = np.array([0, 50])
		self.clip_dir = np.array([0, 0])
		self.clip_plane = np.array([0, 0])
		self.angle = 0
		self.width = 100
		self.height = 100
		self.horizon_y = 50
		self.v_rays = np.array([])

	def move_to(self, x, y):
		self.pos[0] = x
		self.pos[1] = y

	def rotate_to(self, angle):
		self.angle = angle
		rad = np.deg2rad(angle)
		rot = np.array([
			[np.cos(rad), -np.sin(rad)],
			[np.sin(rad), np.cos(rad)]
		])
		self.dir = np.dot(rot, self.dir)
		self.plane = np.dot(rot, self.plane)
		self.clip_dir = np.dot(rot, self.clip_dir)
		self.clip_plane = np.dot(rot, self.clip_plane)

	def move_by(self, dx, dy):
		#self.pos = np.add(self.pos, [dx, dy])
		self.pos[0] += dx
		self.pos[1] += dy

	def rotate_by(self, da):
		self.angle += da
		rad = np.deg2rad(da)
		rot = np.array([
			[np.cos(rad), -np.sin(rad)],
			[np.sin(rad), np.cos(rad)]
		])
		self.dir = np.dot(rot, self.dir)
		self.plane = np.dot(rot, self.plane)
		self.clip_dir = np.dot(rot, self.clip_dir)
		self.clip_plane = np.dot(rot, self.clip_plane)

		for i in range(0, len(self.v_rays)):
			self.v_rays[i] = np.dot(rot, self.v_rays[i])

	def move_forward(self, distance):
		step = self.dir / np.linalg.norm(self.dir) * distance
		self.move_by(step[0], step[1])

	def tilt_by(self, distance):
		self.horizon_y += distance

	def set_fov(self, angle, plane_width, plane_height, clip_width):
		self.width = float(plane_width)
		self.height = float(plane_height)
		self.horizon_y = float(plane_height / 2)

		self.dir = np.array([(plane_width / 2) / np.tan(np.deg2rad(angle / 2)), 0])
		self.plane = np.array([0, plane_width / 2])
		self.clip_dir = np.array([(clip_width / 2) / np.tan(np.deg2rad(angle / 2)), 0])
		self.clip_plane = np.array([0, clip_width / 2])
		self.v_rays = self.generate_rays()

	def generate_rays(self):
		rays = []
		unit_plane = self.plane / math.sqrt(self.plane[0]**2 + self.plane[1]**2)
		for i in range(int(-self.width / 2), int(self.width / 2)):
			plane_pt = self.dir + (unit_plane * i)
			unit_ray = plane_pt / math.sqrt(plane_pt[0]**2 + plane_pt[1]**2)
			rays.append(unit_ray)
		return np.array(rays)

	def rays(self):
		return self.v_rays

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
	if delt_y > 3000:
		y = 2999
	if delt_x > 3000:
		x = 2999

	yield origin, None

	while x < 3000 or y < 3000:
		if x < y:
			hit = origin + dir * abs(x)
			side = 1
			x += delt_x
		else:
			hit = origin + dir * abs(y)
			side = 0
			y += delt_y
		yield hit, side

		# if (int(hit[0]) + 1) % 64 <= 1.0 and (int(hit[1]) + 1) % 64 <= 1.0:
		# 	if int(hit[0] / 64) != int(origin[0] / 64) and int(hit[1] / 64) != int(origin[1] / 64):
		# 		if tilemap.get_tile_px(hit[0] - 1, hit[1] - 1) != 0 or tilemap.get_tile_px(hit[0] + 1, hit[1] - 1) != 0 or tilemap.get_tile_px(hit[0] - 1, hit[1] + 1) != 0 or tilemap.get_tile_px(hit[0] + 1, hit[1] + 1) != 0:
		# 			collisions.append(hit)
		# 			break
		# collisions.append(hit)

def project_point(pt, camera, y_sign=1):
	vector = pt - camera.pos
	proj = camera.dir * (np.dot(vector, camera.dir) / np.dot(camera.dir, camera.dir))
	rej = vector - proj
	proj_len = np.linalg.norm(proj)
	rej_len = np.linalg.norm(rej)
	scaled_rej_len = np.linalg.norm(camera.dir) / proj_len * rej_len
	scaled_rej = rej / rej_len * scaled_rej_len

	if proj_len == 0 or rej_len == 0:
		return None

	# angle = np.deg2rad(camera.angle) - np.arctan2(vector[Y], vector[X])
	# height_factor = 64 * np.linalg.norm(camera.dir)
	# height = height_factor / (np.linalg.norm(vector) * math.cos(angle))
	x_sign = np.sign(np.dot((camera.plane + scaled_rej), camera.plane))
	x = x_sign * np.linalg.norm(camera.plane + scaled_rej)
	height = 16 * np.linalg.norm(camera.dir) / (proj_len)
	y = camera.horizon_y + (y_sign * height)

	return [(x / camera.width - 0.5) * 2, (y / camera.height - 0.5) * 2]

def clip_floor(ul, ur, br, bl, camera):
	edges = [
		[ul, bl],
		[ur, br],
		[ul, ur],
		[bl, br]
		]
	cam_l_ray = camera.v_rays[0]
	cam_r_ray = camera.v_rays[-1]

	final_pts = []

	#filter points that are outside the view frustrum
	for pt in [ul, ur, bl, br]:
		if not point_in_rect(pt, ul, br) or point_in_triangle(pt, camera.pos + (camera.clip_dir * 0.99) - (camera.clip_plane * 1.01), camera.pos + (camera.clip_dir * 0.99) + (camera.clip_plane * 0.99), camera.pos):
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
		[camera.pos + camera.clip_dir - camera.clip_plane, cam_l_ray / np.linalg.norm(cam_l_ray)],
		[camera.pos + camera.clip_dir + camera.clip_plane, cam_r_ray / np.linalg.norm(cam_r_ray)]
		]
	for clip_ray in clip_rays:
		for edge in edges:
			pt = intersect_ray_segment(clip_ray[0], clip_ray[1], edge[0], edge[1])
			if pt is not None:
				final_pts.append(pt)

	#add points where tile edges intersect near clip segment
	clip_segs = [
		[camera.pos + camera.clip_dir - camera.clip_plane, camera.pos + camera.clip_dir + camera.clip_plane]
		]
	for clip_seg in clip_segs:
		for edge in edges:
			pt = intersect_segments(edge[0], edge[1], clip_seg[0], clip_seg[1])
			if pt is not None and point_in_rect(pt, ul, br):
				final_pts.append(pt)

	#add left and right endpoints of near clip segment if they are inside the tile
	near_clip_l = camera.pos + camera.clip_dir - camera.clip_plane
	if point_in_rect(near_clip_l, ul, br):
		final_pts.append(near_clip_l)
	near_clip_r = camera.pos + camera.clip_dir + camera.clip_plane
	if point_in_rect(near_clip_r, ul, br):
		final_pts.append(near_clip_r)

	#sort points
	final_pts = sort_cw(final_pts)

	return final_pts

def clip_wall(left, right, camera):
	cam_l_ray = camera.v_rays[0]
	cam_r_ray = camera.v_rays[-1]

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
	rays = camera.rays()
	max_height = 1.0

	for ray in rays:
		stop = False
		cur_height = 0
		for collision, side in dda(camera.pos, ray, 64, tilemap):
			if stop:
				stop = False
				break

			tile = tilemap.get_tile_px(collision[X], collision[Y])
			tile_coords = [int(collision[X]) / 64, int(collision[Y]) / 64]
			tile_height = tile["floor_height"]

			changed = False
			if tile_height > cur_height:
				cur_height = tile_height
				changed = True
			if cur_height >= max_height:
				stop = True

			if (int(collision[X]) + 1) % 64 <= 1.0 and (int(collision[Y]) + 1) % 64 <= 1.0:
				continue

			if not changed:
				key = (tile_coords[X], tile_coords[Y])
			else:
				key = (tile_coords[X], tile_coords[Y], side)
			if key in used_tiles:
				continue
			used_tiles.add(key)

			if not stop:
				ul = np.array([tile_coords[X] * 64, tile_coords[Y] * 64])
				ur = np.array([tile_coords[X] * 64 + 64, tile_coords[Y] * 64])
				br = np.array([tile_coords[X] * 64 + 64, tile_coords[Y] * 64 + 64])
				bl = np.array([tile_coords[X] * 64, tile_coords[Y] * 64 + 64])
				clip_pts = clip_floor(ul, ur, br, bl, camera)
				floor_pts.append((clip_pts, [ul, br], tile, 0))
			if changed:
				if side == 0:
					y = int(collision[Y] + 32 - (collision[Y] + 32) % 64)
					left = np.array([tile_coords[X] * 64, y])
					right = np.array([tile_coords[X] * 64 + 64, y])
				else:
					x = int(collision[X] + 32 - (collision[X] + 32) % 64)
					left = np.array([x, tile_coords[Y] * 64])
					right = np.array([x, tile_coords[Y] * 64 + 64])
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
				d_mid = np.array([camera.width / 2 - tri_quad[3][X], camera.height / 2 - tri_quad[3][Y]])
				for pt in tri_quad:
					d = pt + d_mid
					og_pts.append([(d[X] / camera.width - 0.5) * 2, (d[Y] / camera.height - 0.5) * 2])
				og_pts = np.array(og_pts)

				trans_pts = []
				for pt in tri_quad:
					trans_pt = project_point(pt, camera, 1)
					if trans_pt is not None:
						height = trans_pt[Y] * 2 - ((camera.horizon_y / camera.height - 0.5) * 2) * 2
						trans_pt[Y] = trans_pt[Y] - height * tile["floor_height"]
						trans_pts.append(trans_pt)
				trans_pts = np.array(trans_pts)

				trans_mid_pt = np.average(trans_pts[0:3], axis=0)
				trans_pts -= trans_mid_pt

				offsets = np.array([
					[(tri_quad[0][X] - tile_x) * recp_tile_w, (tri_quad[0][Y] - tile_y) * recp_tile_h],
					[(tri_quad[1][X] - tile_x) * recp_tile_w, (tri_quad[1][Y] - tile_y) * recp_tile_h],
					[(tri_quad[2][X] - tile_x) * recp_tile_w, (tri_quad[2][Y] - tile_y) * recp_tile_h],
					[(tri_quad[3][X] - tile_x) * recp_tile_w, (tri_quad[3][Y] - tile_y) * recp_tile_h],
					])

				if len(trans_pts) == 4:
					final_quads.append((og_pts, trans_pts, trans_mid_pt, offsets, tile))
		elif surface_type == 1: #wall
			#temp - append tri midpoints
			clip_w = abs(tile_pt[0][1] - tile_pt[0][0])
			tile_pt[0].append(tile_pt[0][0] + clip_w * 1/3.)
			tile_pt[0].append(tile_pt[0][1] - clip_w * 1/3.)

			trans_pts = [] #[ul, bl, ur, br]
			for pt in tile_pt[0]:
				bottom_pt = project_point(pt, camera, 1)
				if bottom_pt is not None:
					height = bottom_pt[Y] * 2 - ((camera.horizon_y / camera.height - 0.5) * 2) * 2
					top_pt = np.array([bottom_pt[X], bottom_pt[Y] - height * tile["floor_height"]])
					trans_pts.append(top_pt)
					trans_pts.append(bottom_pt)

			#get offsets relative to tile width
			recp_tile_w = 1. / np.linalg.norm(tile_pt[1][1] - tile_pt[1][0])
			offset_l = np.linalg.norm(tile_pt[0][0] - tile_pt[1][0]) * recp_tile_w
			offset_r = np.linalg.norm(tile_pt[0][1] - tile_pt[1][0]) * recp_tile_w

			if len(trans_pts) != 8:
				continue

			trans_pts = np.array(trans_pts)

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
			o_l = ((camera.width / 2. - w / 2.) / camera.width - 0.5) * 2
			o_r = ((camera.width / 2. + w / 2.) / camera.width - 0.5) * 2
			o_t = ((camera.height / 2. - 32) / camera.height - 0.5) * 2
			o_b = ((camera.height / 2. + 32) / camera.height - 0.5) * 2
			o_ul = [o_l, o_t]
			o_ur = [o_r, o_t]
			o_br = [o_r, o_b]
			o_bl = [o_l, o_b]

			l_third = abs(trans_pts[1] - trans_pts[0]) * 1/3.
			r_third = abs(trans_pts[3] - trans_pts[2]) * 1/3.

			#triquad 1
			og_pts = np.array([
				o_ul,
				o_br,
				o_bl,
				np.average([o_ul, o_br, o_bl], axis=0)
				])
			tr_pts = np.array([
				trans_pts[0],
				trans_pts[3],
				trans_pts[1],
				intersect_segments(trans_pts[1] - l_third , trans_pts[3] - r_third, trans_pts[4], trans_pts[5])
				])
			mid = np.average(tr_pts[0:3], axis=0)
			tr_pts -= mid
			offs = np.array([
				[offset_l, 1.0],
				[offset_r, 0.0],
				[offset_l, 0.0],
				[(offset_l + offset_r + offset_l) / 3., 1/3.],
				])
			final_quads.append((og_pts, tr_pts, mid, offs, tile))

			#triquad 2
			og_pts = np.array([
				o_ul,
				o_ur,
				o_br,
				np.average([o_ul, o_ur, o_br], axis=0)
				])
			tr_pts = np.array([
				trans_pts[0],
				trans_pts[2],
				trans_pts[3],
				intersect_segments(trans_pts[0] + l_third, trans_pts[2] + r_third, trans_pts[6], trans_pts[7])
				])
			mid = np.average(tr_pts[0:3], axis=0)
			tr_pts -= mid
			offs = np.array([
				[offset_l, 1.0],
				[offset_r, 1.0],
				[offset_r, 0.0],
				[(offset_l + offset_r + offset_r) / 3., 2/3.],
				])
			final_quads.append((og_pts, tr_pts, mid, offs, tile))

	return final_quads