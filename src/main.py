import sys
from tilemap import *
from camera import Camera

def run_software():
	import software
	software.run(tilemap, camera)

def run_gl():
	import opengl
	opengl.run(tilemap, camera)

palette = Palette()
palette.add(0, SimpleTile(  0,   0, 8, 2))
palette.add(1, SimpleTile(  0,  64, 8, 2))
palette.add(2, SimpleTile(-64,   0, 8, 7))
palette.add(3, SimpleTile(  0,  32, 8, 7))
palette.add(4, SimpleTile(-64, 192, 8, 2))
palette.add(5, SimpleTile(-64,  64, 8, 2))

tilemap = TileMap(8, 8, 64)
tilemap.set_tiles_from_palette(palette,
   [[1,1,1,1,1,1,1,1],
	[1,0,3,4,4,0,0,1],
	[1,3,5,2,2,5,0,1],
	[1,0,5,2,2,5,0,1],
	[1,0,0,5,5,0,1,1],
	[1,0,0,0,0,0,0,1],
	[1,0,0,0,0,0,0,1],
	[1,1,1,1,1,1,1,1]])
tilemap.wall_groups[2,4][RIGHT] = Wall(-64, 64, 2)
tilemap.wall_groups[2,5][LEFT] = Wall(0, 32, 5)

camera = Camera()
camera.move_to(224, 288, 32)
camera.set_fov(60, 16, 1000, 120, 80)

if len(sys.argv) > 1:
	renderer = sys.argv[1]
	if renderer == "software":
		run_software()
	elif renderer == "opengl":
		run_gl()
else:
	run_software()