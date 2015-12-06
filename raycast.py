import os
os.environ["PYSDL2_DLL_PATH"] = "C:\\dev\\raycast"
import math
import numpy as np
import sdl2
import sdl2.ext

PALETTE = {}
PALETTE[0] = 0x00000000
PALETTE[1] = 0xFFFFFFFF
PALETTE[2] = 0xFF0000FF
PALETTE[3] = 0x00FF0000
PALETTE[4] = 0x0000FF00
PALETTE[-1] = 0x80808080
PALETTE[-2] = 0x8080FF80

class TilePalette:
	def __init__(self):
		self.palette = {}

	def add(self, index, color):
		self.palette[str(index)] = color

	def get(self, index):
		return self.palette[str(index)]

class TileMap:
	def __init__(self, tiles, size):
		self.tiles = tiles
		self.size = size
		self.width = len(tiles[0])
		self.height = len(tiles)

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
		self.pos = np.array([0, 0])
		self.dir = np.array([100, 0])
		self.plane = np.array([0, 100])
		self.angle = 0
		self.v_rays = []

		self.v_dot = np.vectorize(np.dot)

	def move_to(self, x, y):
		self.pos = [x, y]

	def rotate_to(self, angle):
		self.angle = angle
		rad = np.deg2rad(angle)
		rot = np.array([
			[np.cos(rad), -np.sin(rad)],
			[np.sin(rad), np.cos(rad)]
		])
		self.dir = np.dot(rot, self.dir)
		self.plane = np.dot(rot, self.plane)

	def move_by(self, dx, dy):
		self.pos = np.add(self.pos, [dx, dy])

	def rotate_by(self, da):
		self.angle += da
		rad = np.deg2rad(da)
		rot = np.array([
			[np.cos(rad), -np.sin(rad)],
			[np.sin(rad), np.cos(rad)]
		])
		self.dir = np.dot(rot, self.dir)
		self.plane = np.dot(rot, self.plane)

		#self.v_dot(self.v_rays, rot)
		#for ray in self.v_rays:
		#	ray = np.dot(rot, ray)
		self.v_rays = self.generate_rays()

	def move_forward(self, distance):
		step = self.dir / np.linalg.norm(self.dir) * distance
		self.move_by(step[0], step[1])

	def set_fov(self, angle, width):
		self.width = width
		self.dir = np.array([width / 2 * np.sin(np.deg2rad(angle)), 0])
		self.plane = np.array([0, width])
		self.v_rays = self.generate_rays()
		self.rotate_to(self.angle)

	def generate_rays(self):
		#rays = np.zeros([self.width, 2])
		rays = []
		unit_plane = self.plane / np.sqrt(np.power(self.plane[0], 2) + np.power(self.plane[1], 2))
		for i in range(-self.width / 2, self.width / 2):
			plane_pt = self.dir + (unit_plane * i)
			unit_pt = plane_pt / np.sqrt(np.power(plane_pt[0], 2) + np.power(plane_pt[1], 2))
			rays.append(unit_pt)
		return rays

	def rays(self):
		return self.v_rays
		"""unit_plane = self.plane / math.sqrt(math.pow(self.plane[0], 2) + math.pow(self.plane[1], 2))
		for i in range(-self.width / 2, self.width / 2):
			plane_pt = self.dir + (unit_plane * i)
			unit_pt = plane_pt / math.sqrt(math.pow(plane_pt[0], 2) + math.pow(plane_pt[1], 2))
			yield (self.pos, unit_pt)"""

def dda(p0, p1):
	dp = p1 - p0
	m = abs(dp[1] / dp[0])
	x = p0[0]
	y = p0[1]

	if abs(dp[0]) >= abs(dp[1]):
		i_range = range(abs(int(dp[0])))
		if p1[0] >= p0[0]:
			if p1[1] <= p0[1]: #1
				dx = 1
				dy = -m
			else: #4
				dx = 1
				dy = m
		else:
			if p1[1] <= p0[1]: #2
				dx = -1
				dy = -m
			else: #3
				dx = -1
				dy = m
	else:
		i_range = range(abs(int(dp[1])))
		if p1[0] >= p0[0]:
			if p1[1] <= p0[1]: #1
				dx = 1 / m
				dy = -1
			else: #4
				dx = 1 / m
				dy = 1
		else:
			if p1[1] <= p0[1]: #2
				dx = -1 / m
				dy = -1
			else: #3
				dx = -1 / m
				dy = 1

	for i in i_range:
		x += dx
		y += dy
		yield np.array([x, y])

def dda3(pt, dir, step):
	m = dir[1] / dir[0]
	x = pt[0]
	y = pt[1]

	if abs(dir[0]) >= abs(dir[1]):
		if dir[0] >= 0:
			if dir[1] < 0: #1
				dx = 1
				dy = m
			else: #4
				dx = 1
				dy = m
		else:
			if dir[1] < 0: #2
				dx = -1
				dy = -m
			else: #3
				dx = -1
				dy = -m
	else:
		if dir[0] >= 0:
			if dir[1] < 0: #1
				dx = -1 / m
				dy = -1
			else: #4
				dx = 1 / m
				dy = 1
		else:
			if dir[1] < 0: #2
				dx = -1 / m
				dy = -1
			else: #3
				dx = 1 / m
				dy = 1

	while x >= 0 and x < 384 and y >= 0 and y < 384:
		x += dx
		y += dy
		yield np.array([x, y])

def dda4(pt, dir, step):
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

	delt_x = math.sqrt(math.pow(step, 2) + math.pow(m * step, 2))
	delt_y = math.sqrt(math.pow(step, 2) + math.pow(1/m * step, 2))

	if dx > 0:
		side_x = abs(step - pt[0] % step) * delt_x / step + 1
	else:
		side_x = abs(pt[0] % step) * delt_x / step + 1

	if dy > 0:
		side_y = abs(step - pt[1] % step) * delt_y / step + 1
	else:
		side_y = abs(pt[1] % step) * delt_y / step + 1

	x = side_x
	y = side_y
	if delt_y > 1000:
		y = 999
	if delt_x > 1000:
		x = 999

	yield 1

	#while x >= 0 and x < 384 and y >= 0 and y < 384:
	#while abs(x) < 1000 and abs(y) < 1000:
	while x < 1000 or y < 1000:
		if x < y:
			yield abs(x)
			x += delt_x
		else:
			yield abs(y)
			y += delt_y

def get_collision_point(tilemap, ray):
	for pt in dda(ray[0], ray[0] + ray[1] * 200):
		if tilemap.get_tile_px(pt[0], pt[1]) > 0:
			return pt
	return None

def get_collision_point2(tilemap, ray):
	for pt in dda3(ray[0], ray[1], 400):
		if tilemap.get_tile_px(pt[0], pt[1]) > 0:
			return pt
	return None

def get_collision_point4(tilemap, ray):
	for pt in dda4(ray[0], ray[1], 64):
		tile = tilemap.get_tile_px(pt[0], pt[1])
		if tile > 0:
			return pt, tile
	return None, None

def render_grid(tilemap, color):
	for x in range(0, tilemap.width):
		renderer.draw_line((x * tilemap.size, 0, x * tilemap.size, tilemap.height * tilemap.size), color)

	for y in range(0, tilemap.height):
		renderer.draw_line((0, y * tilemap.size, tilemap.width * tilemap.size, y * tilemap.size), color)

def render_tiles(tilemap):
	for y in range(0, tilemap.height):
		for x in range(0, tilemap.width):
			tile = tilemap.get_tile(x, y)
			if tile != 0:
				renderer.fill((x * tilemap.size, y * tilemap.size, tilemap.size, tilemap.size), PALETTE[tile])

def render_camera(camera):
	renderer.fill((int(camera.pos[0] - 8), int(camera.pos[1] - 8), 16, 16), 0xFFF0000)
	renderer.draw_line((int(camera.pos[0]), int(camera.pos[1]), int(camera.pos[0] + camera.dir[0]), int(camera.pos[1] + camera.dir[1])), 0xFFF0000)

	plane_p0 = camera.pos + camera.dir - camera.plane / 2
	plane_p1 = camera.pos + camera.dir + camera.plane / 2

	renderer.draw_line((int(plane_p0[0]), int(plane_p0[1]), int(plane_p1[0]), int(plane_p1[1])), 0xFFF0000)
	renderer.draw_line((int(camera.pos[0]), int(camera.pos[1]), int(plane_p0[0]), int(plane_p0[1])), 0xFFF0000)
	renderer.draw_line((int(camera.pos[0]), int(camera.pos[1]), int(plane_p1[0]), int(plane_p1[1])), 0xFFF0000)

def render_scan(camera, tilemap):
	scan_line_x = 0

	for v_ray in camera.rays():
		ray = (camera.pos, v_ray)

		#renderer.draw_line((int(ray[0][0]), int(ray[0][1]), int(ray[1][0]), int(ray[1][1])), 0xFFF0000)
		#collision, tile = get_collision_point4(tilemap, ray)

		floors = []
		collision = None
		tile = None
		for dist in dda4(ray[0], ray[1], 64):
			pt = ray[0] + ray[1] * dist
			tile = tilemap.get_tile_px(pt[0], pt[1])
			f_pt = ray[0] + ray[1] * dist
			f_tile = tilemap.get_tile_px(f_pt[0], f_pt[1])
			if tile > 0:
				collision = pt
				break
			else:
				floors.append((f_pt, f_tile))

		if collision != None:
			#renderer.fill((int(collision[0] - 2), int(collision[1] - 2), 4, 4), 0xFF0000FF)
			angle = np.deg2rad(camera.angle) - np.arctan2(ray[1][1], ray[1][0])

			for f in floors:
				f_height = 64 * np.linalg.norm(camera.dir) / (np.linalg.norm(f[0] - camera.pos) * np.cos(angle))
				if (np.abs(f_height) < 1000):
					renderer.fill((400 + scan_line_x, 240, 2, int(f_height / 2)), PALETTE[f[1]])

			height = 64 * np.linalg.norm(camera.dir) / (np.linalg.norm(collision - camera.pos) * np.cos(angle))
			if (np.abs(height) < 1000):
				renderer.fill((400 + scan_line_x, 240 - int(height / 2), 2, int(height)), PALETTE[tile])

		scan_line_x += 2

window = sdl2.ext.Window("Hello World!", size=(1280, 480))
window.show()

surface = window.get_surface()

renderer = sdl2.ext.Renderer(surface, flags=sdl2.SDL_RENDERER_ACCELERATED)

#processor = sdl2.ext.TestEventProcessor()
#processor.run(window)

tilemap = TileMap([
	[0,4,4,1,1,0],
	[2,0,0,0,1,0],
	[2,0,1,0,0,0],
	[2,-1,-2,1,0,0],
	[2,-2,-1,1,0,0],
	[1,3,3,1,0,1]], 64)

camera = Camera()
camera.move_to(160, 224)
camera.set_fov(60, 480)

grid_color = sdl2.ext.Color(0, 255, 0, 255)
floor_color = sdl2.ext.Color(0, 96, 96, 128)

running = True
while running:
	now = sdl2.timer.SDL_GetTicks()

	#update
	events = sdl2.ext.get_events()
	for event in events:
		if event.type == sdl2.SDL_QUIT:
			running = False
		elif event.type == sdl2.SDL_KEYDOWN:
			if event.key.keysym.sym == sdl2.SDLK_ESCAPE:
				running = False
			if event.key.keysym.sym == sdl2.SDLK_UP:
				camera.move_forward(4)
			if event.key.keysym.sym == sdl2.SDLK_DOWN:
				camera.move_forward(-4)
			if event.key.keysym.sym == sdl2.SDLK_LEFT:
				camera.rotate_by(-5)
			if event.key.keysym.sym == sdl2.SDLK_RIGHT:
				camera.rotate_by(5)

	#render
	renderer.clear(0xFF000000)
	render_grid(tilemap, grid_color)
	render_tiles(tilemap)
	render_camera(camera)
	render_scan(camera, tilemap)

	renderer.present()

	delay = 1000 / 60 - (sdl2.timer.SDL_GetTicks() - now)
	print 1000 / (sdl2.timer.SDL_GetTicks() - now)
	if delay > 0:
		sdl2.timer.SDL_Delay(delay)

	window.refresh()

window.hide()