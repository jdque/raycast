import os
os.environ["PYSDL2_DLL_PATH"] = os.path.dirname(__file__) + os.path.sep + ".." + os.path.sep + "lib"

import sdl2
import sdl2.ext
import timeit

from engine import *

window = sdl2.ext.Window("Hello World!", size=(1600, 960))
surface = window.get_surface()
renderer = sdl2.ext.Renderer(surface, flags=sdl2.SDL_RENDERER_ACCELERATED)
render_plane = (464, 0, 960, 640)

def render_grid(tilemap):
	for x in range(0, tilemap.width):
		renderer.draw_line((x * tilemap.size, 0, x * tilemap.size, tilemap.height * tilemap.size), 0xFF00FF00)

	for y in range(0, tilemap.height):
		renderer.draw_line((0, y * tilemap.size, tilemap.width * tilemap.size, y * tilemap.size), 0xFF00FF00)

def render_tiles(tilemap):
	for y in range(0, tilemap.height):
		for x in range(0, tilemap.width):
			tile = tilemap.get_floor(x, y)
			z = tile.z + tile.height
			if z == 0:
				color = 0xFF000000
			elif z > 0:
				color = 0xFFFFFFFF
			elif z < 0:
				color = 0xFF808080
			renderer.fill((x * tilemap.size, y * tilemap.size, tilemap.size, tilemap.size), color)

def render_camera(camera):
	renderer.fill((int(camera.pos[0] - 1), int(camera.pos[1] - 1), 2, 2), 0xFFF0000)
	renderer.draw_line((int(camera.pos[0]), int(camera.pos[1]), int(camera.pos[0] + camera.near_dir[0]), int(camera.pos[1] + camera.near_dir[1])), 0xFFF0000)

	plane_p0 = camera.pos + camera.near_dir - camera.near_plane
	plane_p1 = camera.pos + camera.near_dir + camera.near_plane

	renderer.draw_line((int(plane_p0[0]), int(plane_p0[1]), int(plane_p1[0]), int(plane_p1[1])), 0xFFF0000)
	renderer.draw_line((int(camera.pos[0]), int(camera.pos[1]), int(plane_p0[0]), int(plane_p0[1])), 0xFFF0000)
	renderer.draw_line((int(camera.pos[0]), int(camera.pos[1]), int(plane_p1[0]), int(plane_p1[1])), 0xFFF0000)

def render_border(plane):
	renderer.draw_line((int(plane[X]), int(plane[Y]), int(plane[X] + plane[W]), int(plane[Y])), 0xFFF0000)
	renderer.draw_line((int(plane[X]), int(plane[Y]), int(plane[X]), int(plane[Y] + plane[H])), 0xFFF0000)
	renderer.draw_line((int(plane[X] + plane[W]), int(plane[Y] + plane[H]), int(plane[X] + plane[W]), int(plane[Y])), 0xFFF0000)
	renderer.draw_line((int(plane[X] + plane[W]), int(plane[Y] + plane[H]), int(plane[X]), int(plane[Y] + plane[H])), 0xFFF0000)

def render_scan(camera, tilemap):
	floor_pts, wall_pts = get_clipped_tile_points(tilemap, camera)
	for floor_pt in floor_pts:
		tris = triangulate(floor_pt.points)
		for tri in tris:
			renderer.draw_line((int(tri[0][X]), int(tri[0][Y]), int(tri[1][X]), int(tri[1][Y])), 0x8000FFFF)
			renderer.draw_line((int(tri[0][X]), int(tri[0][Y]), int(tri[2][X]), int(tri[2][Y])), 0x8000FFFF)
			renderer.draw_line((int(tri[1][X]), int(tri[1][Y]), int(tri[2][X]), int(tri[2][Y])), 0x8000FFFF)
			mid = np.average(tri[0:3], axis=0)
			renderer.fill((int(mid[X]) - 2, int(mid[Y]) - 2, 4, 4), 0x8000FFFF)
	for wall_pt in wall_pts:
		renderer.draw_line((int(wall_pt.points[0][X]), int(wall_pt.points[0][Y]), int(wall_pt.points[1][X]), int(wall_pt.points[1][Y])), 0xFFFF00FF)

	quads = get_tri_quads(floor_pts + wall_pts, camera)
	for quad in quads:
		q = quad[1] + quad[2]
		qt = (q * 0.5 + 0.5) * [render_plane[W], render_plane[H], 1] + [render_plane[X], render_plane[Y], 0]
		renderer.draw_line((int(qt[0][X]), int(qt[0][Y]), int(qt[1][X]), int(qt[1][Y])), 0xFFFFFFFF)
		renderer.draw_line((int(qt[0][X]), int(qt[0][Y]), int(qt[2][X]), int(qt[2][Y])), 0xFFFFFFFF)
		renderer.draw_line((int(qt[1][X]), int(qt[1][Y]), int(qt[2][X]), int(qt[2][Y])), 0xFFFFFFFF)
		renderer.fill((int(qt[3][X]) - 2, int(qt[3][Y]) - 2, 4, 4), 0xFFFFFFFF)

def run(tilemap, camera):
	# texture_surface = sdl2.ext.load_image("C:\\dev\\raycast\\textures.png")
	# factory = sdl2.ext.SpriteFactory(sprite_type=sdl2.ext.TEXTURE, renderer=renderer)
	# texture_sprite = factory.from_surface(texture_surface)

	window.show()

	running = True
	while running:
		now = sdl2.timer.SDL_GetTicks()

		#update
		events = sdl2.ext.get_events()
		for event in events:
			if event.type == sdl2.SDL_QUIT:
				running = False

		key_states = sdl2.keyboard.SDL_GetKeyboardState(None)
		if key_states[sdl2.SDL_SCANCODE_ESCAPE]:
			running = False
		if key_states[sdl2.SDL_SCANCODE_UP]:
			camera.move_forward(4)
		elif key_states[sdl2.SDL_SCANCODE_DOWN]:
			camera.move_forward(-4)
		if key_states[sdl2.SDL_SCANCODE_LEFT]:
			camera.rotate_by(-5)
		elif key_states[sdl2.SDL_SCANCODE_RIGHT]:
			camera.rotate_by(5)
		if key_states[sdl2.SDL_SCANCODE_W]:
			camera.move_by(0, 0, 4)
		elif key_states[sdl2.SDL_SCANCODE_S]:
			camera.move_by(0, 0, -4)

		#render
		renderer.clear(0xFF000000)
		render_tiles(tilemap)
		render_grid(tilemap)
		render_camera(camera)
		render_scan(camera, tilemap)
		render_border(render_plane)
		renderer.present()

		delay = 1000 / 60 - (sdl2.timer.SDL_GetTicks() - now)
		#print 1000 / (sdl2.timer.SDL_GetTicks() - now)
		if delay > 0:
			sdl2.timer.SDL_Delay(delay)

		window.refresh()

	window.hide()

def run_test():
	palette = TilePalette()
	palette.add(0, Tile(0, 0, 0, 8, 2))
	palette.add(1, Tile(1, 64, 0, 8, 2))
	palette.add(2, Tile(1, 0, -64, 8, 7))
	palette.add(3, Tile(1, 32, 0, 8, 7))
	palette.add(4, Tile(1, 192, -64, 8, 2))
	palette.add(5, Tile(1, 64, -64, 8, 2))

	tilemap = TileMap(7, 7, 64)
	tilemap.set_tiles_from_palette(palette,
	   [[1,1,1,1,1,1,1],
		[1,0,3,4,4,0,1],
		[1,3,5,2,2,5,1],
		[1,0,5,2,2,5,1],
		[1,0,0,5,5,0,1],
		[1,0,0,0,0,0,1],
		[1,1,1,1,1,1,1]])

	camera = Camera()
	camera.move_to(224, 288, 32)
	camera.set_fov(60, 32, 1000, 480, 320)

	def do():
		#for i in r:
		#	dda(pos, ray, 64)
		#intersect_segments(a1, a2, b1, b2)
		floor_pts, wall_pts = get_clipped_tile_points(tilemap, camera)
		get_tri_quads(floor_pts + wall_pts, camera)

	print timeit.timeit(do, number=100)
	#print timeit.timeit(do2, number=30)