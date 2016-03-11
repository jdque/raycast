import os
os.environ["PYSDL2_DLL_PATH"] = "C:\\dev\\raycast"
import sys
sys.path.append("C:\\dev\\raycast")
import numpy as np

import sdl2
import sdl2.ext
import sdl2.video

import OpenGL.GL as GL
from OpenGL.GL import shaders

from engine import *

window = None
surface = None
renderer = None
context = None
shaderProgram = None
textures = []
running = True

test_tris = []

VAO_VERTEX = -1
ATTR_POSITION = -1
ATTR_TEXCOORDS = -1
ATTR_TEXUNIT = -1
ATTR_MODELVIEWMAT = -1
ATTR_VERTPROJMAT = -1
ATTR_TEXPROJMAT = -1
ATTR_TEXBOUNDS = -1

def render_raycast(camera, tilemap):
	global test_tris

	test_tris = []
	floor_pts, wall_pts = get_clipped_tile_points(tilemap, camera)
	quads = get_tri_quads(wall_pts + floor_pts, camera)
	for quad in quads:
		pos = quad[2]
		o_tri = np.c_[quad[0], np.zeros(4)]
		t_tri = np.c_[quad[1], np.zeros(4)]
		offset = np.c_[quad[3], np.zeros(4)]

		o_tri[:,1] *= -1
		t_tri[:,1] *= -1
		pos[1] *= -1

		tile_kind = quad[4]["kind"]

		if tile_kind == 0:
			tile_tex = quad[4]["floor_tex"]
		elif tile_kind == 1:
			tile_tex = quad[4]["wall_tex"]

		#if tile_kind != 1:
		area = 0.5 * np.linalg.det(np.array([
			[1.0, t_tri[0][0], t_tri[0][1]],
			[1.0, t_tri[1][0], t_tri[1][1]],
			[1.0, t_tri[2][0], t_tri[2][1]],
			]))
		if abs(area) <= 0.00001:
			continue

		test_tris.append(TriQuad(
			o_tri,
			t_tri,
			make_model_view_mat(pos[X], pos[Y], 0.0, 1.0, 1.0, 1.0),
			offset,
			tile_tex))

def update_raycast():
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
		camera.tilt_by(-4)
	elif key_states[sdl2.SDL_SCANCODE_S]:
		camera.tilt_by(4)

render_plane = (0, 0, 960, 640)

palette = TilePalette()
palette.add(0, 0, 0.0, 8, 2)
palette.add(1, 1, 1.0, 8, 2)
palette.add(2, 1, 0.2, 8, 7)

tilemap = TileMap(7, 7, 64)
tilemap.set_tiles_from_palette(palette,
   [[1,1,1,1,1,1,1],
	[1,0,0,1,0,0,1],
	[1,0,2,2,0,0,1],
	[1,0,2,0,0,0,1],
	[1,0,0,0,0,0,1],
	[1,0,0,0,0,0,1],
	[1,1,1,1,1,1,1]])

camera = Camera()
camera.move_to(224, 288)
camera.set_fov(60, 120, 80, 32)

#------------------------------------------------------------------------------------------

class TriQuad:
	def __init__(self, original, transformed, model_view_mat, texture_bounds, texture_id):
		self.model_view_mat = model_view_mat
		self.vert_proj_mat = make_proj_mat(original, transformed)
		self.tex_proj_mat = make_proj_mat(transformed, texture_bounds)

		self.vertices = np.array(original[0:3], dtype=np.float32).flatten()
		self.VBO = GL.glGenBuffers(1)
		GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.VBO)
		GL.glBufferData(GL.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL.GL_DYNAMIC_DRAW)
		GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

		self.texture_id = texture_id

	def __del__(self):
		GL.glDeleteBuffers(1, np.array([self.VBO]))

def make_model_view_mat(x, y, z, sx, sy, sz):
	model_view_mat = np.array([
		[ sx, 0.0, 0.0,  x ],
		[0.0,  sy, 0.0,  y ],
		[0.0, 0.0,  sz,  z ],
		[0.0, 0.0, 0.0, 1.0]
		])
	return model_view_mat

def make_proj_mat(old, new):
	"""
	Using four corner points of quad, solve for the coefficients m1-m8 in system:

	|s|   |m1, m2,  0, m3|   |x|
	|t| = |m4, m5,  0, m6| * |y|
	|0|   | 0,  0,  1,  0|   |0|
	|w|   |m7, m8,  0,  1|   |1|

    x, y  = transformed corner vertices in object coordinates
    s, t  = original corner vertices in object coordinates
    s/w, t/w = projected texture coordinates
	"""

	# old = np.array([
	# 	[-1.0,  1.0, 0.0],
	# 	[ 1.0,  0.3, 0.0],
	# 	[ 1.0, -0.3, 0.0],
	# 	[-1.0, -1.0, 0.0]
	# 	], dtype=np.float32)

	# new = np.array([
	# 	[-1.0,  1.0, 0.0],
	# 	[ 1.0,  1.0, 0.0],
	# 	[ 1.0, -1.0, 0.0],
	# 	[-1.0, -1.0, 0.0]
	# 	], dtype=np.float32)

	coeffs = []
	rhs = []
	for o, n in zip(old, new):
		coeffs.append(np.array([o[0], o[1], 1, 0, 0, 0, -o[0] * n[0], -o[1] * n[0]]))
		coeffs.append(np.array([0, 0, 0, o[0], o[1], 1, -o[0] * n[1], -o[1] * n[1]]))
		rhs.append(n[0])
		rhs.append(n[1])

	X = np.linalg.solve(np.array(coeffs), np.array(rhs))

	proj_mat = np.array([
		[X[0], X[1],  0., X[2]],
		[X[3], X[4],  0., X[5]],
		[  0.,   0.,  1.,   0.],
		[X[6], X[7],  0.,   1.]
		], dtype=np.float32)

	return proj_mat

def init():
	global window
	global surface
	global renderer
	global context
	global shaderProgram

	#SDL
	window = sdl2.ext.Window("Hello World!", size=(960, 640), flags=sdl2.SDL_WINDOW_OPENGL)
	window.show()
	surface = window.get_surface()
	renderer = sdl2.ext.Renderer(surface, flags=sdl2.SDL_RENDERER_ACCELERATED)

	#OpenGL
	#sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_DOUBLEBUFFER, 1)
	#sdl2.video.SDL_GL_SetAttribute(sdl2.video.SDL_GL_CONTEXT_PROFILE_MASK, sdl2.video.SDL_GL_CONTEXT_PROFILE_CORE)
	context = sdl2.SDL_GL_CreateContext(window.window)
	GL.glEnable(GL.GL_DEPTH_TEST)
	GL.glDepthFunc(GL.GL_LESS)
	GL.glClearColor(1.0, 1.0, 1.0, 1.0)

	print GL.glGetString(GL.GL_VERSION)

	#Shaders
	vertexShader = GL.shaders.compileShader("""
		#version 430

		layout (location=0) in vec3 v_position;
		layout (location=1) in vec2 v_texCoords;

		uniform mat4 v_modelViewMat;
		uniform mat4 v_vertProjMat;
		uniform mat4 v_texProjMat;

		noperspective out vec4 f_texCoords;

		void main()
		{
			vec4 projPos = v_vertProjMat * vec4(v_position, 1);

			vec4 texCoords = v_texProjMat * (projPos / projPos.w);
			f_texCoords = vec4(texCoords.x, texCoords.y, texCoords.z, texCoords.w);
			gl_Position = v_modelViewMat * projPos;
		}
		""", GL.GL_VERTEX_SHADER)

	fragmentShader = GL.shaders.compileShader("""
		#version 430

		uniform sampler2D f_texUnit;
		noperspective in vec4 f_texCoords;

		void main()
		{
			//gl_FragColor = vec4(color, 1.0);
			//gl_FragColor = texture(f_texUnit, f_texCoords);
			//vec2 newCoords = vec2(f_texCoords.x / f_texCoords.w, f_texCoords.y / f_texCoords.w);
			gl_FragColor = texture2DProj(f_texUnit, f_texCoords);
			//gl_FragColor = vec4( vec2(f_texCoords.x / f_texCoords.w, f_texCoords.y / f_texCoords.w),0,1);
		}
		""", GL.GL_FRAGMENT_SHADER)

	shaderProgram = GL.shaders.compileProgram(vertexShader, fragmentShader)

def update():
	global running

	events = sdl2.ext.get_events()
	for event in events:
		if event.type == sdl2.SDL_QUIT:
			running = False
		elif event.type == sdl2.SDL_KEYDOWN:
			if event.key.keysym.sym == sdl2.SDLK_ESCAPE:
				running = False

def render():
	#renderer.clear(0xFF000000)
	#renderer.present()
	#window.refresh()

	GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

	GL.glUseProgram(shaderProgram)

	#wireframe
	#GL.glPolygonMode(GL.GL_FRONT, GL.GL_LINE)

	GL.glActiveTexture(GL.GL_TEXTURE0)

	GL.glBindVertexArray(VAO_VERTEX)
	for tri in test_tris:
		GL.glUniformMatrix4fv(ATTR_MODELVIEWMAT, 1, GL.GL_TRUE, tri.model_view_mat)
		GL.glUniformMatrix4fv(ATTR_VERTPROJMAT, 1, GL.GL_TRUE, tri.vert_proj_mat)
		GL.glUniformMatrix4fv(ATTR_TEXPROJMAT, 1, GL.GL_TRUE, tri.tex_proj_mat)
		GL.glBindTexture(GL.GL_TEXTURE_2D, tri.texture_id)
		GL.glBindBuffer(GL.GL_ARRAY_BUFFER, tri.VBO)
		GL.glVertexAttribPointer(ATTR_POSITION, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
		GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(tri.vertices))
		GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
		GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

	GL.glBindVertexArray(0)

	sdl2.SDL_GL_SwapWindow(window.window)

def run():
	global VAO_VERTEX
	global ATTR_POSITION
	global ATTR_TEXCOORDS
	global ATTR_TEXUNIT
	global ATTR_MODELVIEWMAT
	global ATTR_VERTPROJMAT
	global ATTR_TEXPROJMAT
	global ATTR_TEXBOUNDS
	global textures
	global test_quad

	init()

	vertices = np.array([
		-1.0,  1.0, 0.0,
		 1.0,  0.3, 0.0,
		 1.0, -0.3, 0.0,
		-1.0,  1.0, 0.0,
		 1.0, -0.3, 0.0,
		-1.0, -1.0, 0.0
		], dtype=np.float32)

	colors = np.array([
		1.0, 1.0, 1.0,
		0.0, 0.0, 0.0,
		0.0, 0.0, 0.0,
		0.0, 0.0, 0.0,
		1.0, 1.0, 1.0,
		0.0, 0.0, 0.0
		], dtype=np.float32)

	tex_coords = np.array([
		0.0, 1.0,
		1.0, 1.0,
		1.0, 0.0,
		0.0, 1.0,
		1.0, 0.0,
		0.0, 0.0
		], dtype=np.float32)

	ATTR_MODELVIEWMAT = GL.glGetUniformLocation(shaderProgram, "v_modelViewMat")
	ATTR_VERTPROJMAT = GL.glGetUniformLocation(shaderProgram, "v_vertProjMat")
	ATTR_TEXPROJMAT = GL.glGetUniformLocation(shaderProgram, "v_texProjMat")
	ATTR_TEXBOUNDS = GL.glGetUniformLocation(shaderProgram, "v_texBounds")
	ATTR_TEXUNIT = GL.glGetUniformLocation(shaderProgram, "f_texUnit")

	VAO_VERTEX = GL.glGenVertexArrays(1)
	GL.glBindVertexArray(VAO_VERTEX)
	ATTR_POSITION = GL.glGetAttribLocation(shaderProgram, "v_position")
	GL.glEnableVertexAttribArray(ATTR_POSITION)
	GL.glBindVertexArray(0)

	#VBO_positions = GL.glGenBuffers(1)
	#GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO_positions)
	#GL.glBufferData(GL.GL_ARRAY_BUFFER, test_quads[0].VBO.nbytes, test_quads[0].VBO, GL.GL_STATIC_DRAW)

	#VBO_tex_coords = GL.glGenBuffers(1)
	#GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO_tex_coords)
	#GL.glBufferData(GL.GL_ARRAY_BUFFER, tex_coords.nbytes, tex_coords, GL.GL_STATIC_DRAW)

	# ATTR_TEXCOORDS = GL.glGetAttribLocation(shaderProgram, "v_texCoords")
	# GL.glEnableVertexAttribArray(ATTR_TEXCOORDS)
	# GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO_tex_coords)
	# GL.glVertexAttribPointer(ATTR_TEXCOORDS, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

	texture_surface = sdl2.ext.load_image("C:\\dev\\raycast\\textures.png")
	sprite_factory = sdl2.ext.SpriteFactory(sprite_type=sdl2.ext.SOFTWARE, renderer=renderer)
	test_sprite = sprite_factory.create_software_sprite((64, 64))
	for i in range(8):
		sdl2.SDL_BlitSurface(
			texture_surface, sdl2.SDL_Rect(i * 64, 0, 64, 64),
			test_sprite.surface, sdl2.SDL_Rect(0, 0, 64, 64))
		pixels = sdl2.ext.pixels2d(test_sprite)
		pixels = np.copy(pixels)

		texture = GL.glGenTextures(1)
		GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
		GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
		GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, 64, 64, 0, GL.GL_BGRA, GL.GL_UNSIGNED_INT_8_8_8_8_REV, np.flipud(pixels.T).reshape(64*64))
		GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
		GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

		textures.append(texture)

	test_tris.append(TriQuad(
		np.array([
			[-1.0,  1.0, 0.0],
			[ 1.0,  1.0, 0.0],
			[ 1.0, -1.0, 0.0],
			[-1.0, -1.0, 0.0]
			]),
		np.array([
			[-0.4,  0.4, 0.0],
			[ 0.4,  0.12, 0.0],
			[ 0.4, -0.12, 0.0],
			[-0.4, -0.4, 0.0]
			]),
		make_model_view_mat(-0.6, 0.0, 0.0, 1.0, 1.0, 1.0),
		np.array([
			[0.0, 1.0, 0.0],
			[1.0, 1.0, 0.0],
			[1.0, 0.0, 0.0],
			[0.0, 0.0, 0.0]
			]),
		textures[0]))
	test_tris.append(TriQuad(
		np.array([
			[-1.0,  1.0, 0.0],
			[ 1.0,  1.0, 0.0],
			[ 1.0, -1.0, 0.0],
			[-1.0, -1.0, 0.0]
			]),
		np.array([
			[-1.0,  0.3, 0.0],
			[ 1.0,  1.0, 0.0],
			[ 1.0, -1.0, 0.0],
			[-1.0, -0.3, 0.0]
			]),
		make_model_view_mat(0.6, 0.7, 0.0, 0.4, 0.4, 1.0),
		np.array([
			[0.0, 1.0, 0.0],
			[1.0, 1.0, 0.0],
			[1.0, 0.0, 0.0],
			[0.0, 0.0, 0.0]
			]),
		textures[3]))

	while running:
		now = sdl2.timer.SDL_GetTicks()

		update()
		update_raycast()
		render_raycast(camera, tilemap)
		render()

		delay = 1000 / 60 - (sdl2.timer.SDL_GetTicks() - now)
		print 1000 / (sdl2.timer.SDL_GetTicks() - now)
		if delay > 0:
		 	sdl2.timer.SDL_Delay(delay)

	window.hide()
	#sdl2.SDL_GL_DeleteContext(context)
	#sdl2.SDL_DestroyWindow(window.window)
	#sdl2.SDL_Quit()

run()