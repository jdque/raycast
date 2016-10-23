import functools
import math
import numpy as np

def round_down(number, multiple = 1):
	return number - (number % multiple)

def round_up(number, multiple = 1):
	return number - (number % multiple) + multiple

def round_nearest(number, multiple = 1):
	rem = number % multiple
	return (number - rem + multiple) if rem >= multiple / 2 else number - rem

def norm2(vector):
	return math.sqrt(vector[0]**2 + vector[1]**2)

def intersect_ray_segment(ray_o, ray_d, seg_a, seg_b):
	v1 = ray_o - seg_a
	v2 = seg_b - seg_a
	v3 = np.array([-ray_d[1], ray_d[0]])
	t = np.dot(v1, v3) / np.dot(v2, v3)
	if t < 0 or t > 1:
		return None
	pt = seg_a + (seg_b - seg_a) * t
	if np.dot(pt - ray_o, ray_d) < 0:
		return None
	return pt

def intersect_segments(a1, a2, b1, b2):
	da = a2 - a1
	db = b2 - b1
	dp = a1 - b1
	dap = np.array([-da[1], da[0]])
	denom = np.dot(dap, db)
	num = np.dot(dap, dp)
	if denom == 0:
		return None
	div = num / denom
	if div < 0 or div > 1:
		return None
	pt = div * db + b1
	return pt

def point_in_triangle(pt, p0, p1, p2):
	# dx = pt[0] - p2[0]
	# dy = pt[1] - p2[1]
	# dx21 = p2[0] - p1[0]
	# dy12 = p1[1] - p2[1]
	# d = dy12 * (p0[0] - p2[0]) + dx21 * (p0[1] - p2[1])
	# s = dy12 * dx + dx21 * dy
	# t = (p2[1] - p0[1]) * dx + (p0[0] - p2[0]) * dy
	# if d < 0:
	# 	return s <= 0 and t <= 0 and s + t >= d
	# return s >= 0 and t >= 0 and s + t <= d
	a = ((p1[1] - p2[1])*(pt[0] - p2[0]) + (p2[0] - p1[0])*(pt[1] - p2[1])) / ((p1[1] - p2[1])*(p0[0] - p2[0]) + (p2[0] - p1[0])*(p0[1] - p2[1]))
	b = ((p2[1] - p0[1])*(pt[0] - p2[0]) + (p0[0] - p2[0])*(pt[1] - p2[1])) / ((p1[1] - p2[1])*(p0[0] - p2[0]) + (p2[0] - p1[0])*(p0[1] - p2[1]))
	c = 1 - a - b
	if a < 0 or a > 1 or b < 0 or b > 1 or c < 0 or c > 1:
		return False
	return True

def point_in_rect(pt, rect_min_pt, rect_max_pt):
	if pt[0] < rect_min_pt[0] or pt[1] < rect_min_pt[1] or pt[0] > rect_max_pt[0] or pt[1] > rect_max_pt[1]:
		return False
	return True

def sort_cw(pts):
	center = np.average(pts, axis=0)
	def less(a, b):
		da = a - center
		db = b - center
		if da[0] >= 0 and db[0] < 0:
			return -1
		if da[0] < 0 and db[0] >= 0:
			return 1
		if da[0] == 0 and db[0] == 0:
			if da[1] >= 0 or db[1] >= 0:
				return a[1] > b[1]
			return b[1] > a[1]

		det = (da[0]) * (db[1]) - (db[0]) * (da[1])
		if det < 0:
			return -1
		elif det > 0:
			return 1

		return -1 if da[0]**2 + da[1]**2 > db[0]**2 + db[1]**2 else 1
	sorted_pts = sorted(pts, key=functools.cmp_to_key(less))
	return sorted_pts

def triangulate(pts):
	tris = []
	for i in range(1, len(pts) - 1):
		tris.append([pts[0], pts[i], pts[i+1]])
	return tris

def dda(origin, dir, step):
	#TODO - try casting m to int
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

	delt_x = np.sqrt(step**2 + (m * step)**2)
	delt_y = np.sqrt(step**2 + (1/m * step)**2)

	if dx > 0:
		side_x = abs(step - origin[0] % step) * delt_x / step
	else:
		side_x = abs(origin[0] % step) * delt_x / step + 1

	if dy > 0:
		side_y = abs(step - origin[1] % step) * delt_y / step
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