import os
os.environ["PYSDL2_DLL_PATH"] = "C:\\dev\\raycast"
import math
import numpy as np

import sdl2
import sdl2.ext
import sdl2.video

import OpenGL.GL as GL
from OpenGL.GL import shaders

window = None
surface = None
renderer = None
context = None
shaderProgram = None
VAO_VERTEX = None
textures = []
running = True

test_quads = []

ATTR_POSITION = -1
ATTR_TEXCOORDS = -1
ATTR_TEXUNIT = -1
ATTR_MODELVIEWMAT = -1
ATTR_VERTPROJMAT = -1
ATTR_TEXPROJMAT = -1

def init():
	global window
	global surface
	global renderer
	global context
	global shaderProgram

	#SDL
	window = sdl2.ext.Window("Hello World!", size=(640, 640), flags=sdl2.SDL_WINDOW_OPENGL)
	window.show()
	surface = window.get_surface()
	renderer = sdl2.ext.Renderer(surface, flags=sdl2.SDL_RENDERER_ACCELERATED)

	#OpenGL
	#sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_DOUBLEBUFFER, 1)
	#sdl2.video.SDL_GL_SetAttribute(sdl2.video.SDL_GL_CONTEXT_PROFILE_MASK, sdl2.video.SDL_GL_CONTEXT_PROFILE_CORE)
	context = sdl2.SDL_GL_CreateContext(window.window)
	GL.glEnable(GL.GL_DEPTH_TEST)
	GL.glDepthFunc(GL.GL_LESS)
	GL.glClearColor(1.0, 0.0, 0.5, 1.0)

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
			f_texCoords = v_texProjMat * (projPos / projPos.w);
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
	for quad in test_quads:
		GL.glBindTexture(GL.GL_TEXTURE_2D, quad.texture)
		GL.glUniformMatrix4fv(ATTR_MODELVIEWMAT, 1, GL.GL_TRUE, quad.model_view_mat)
		GL.glUniformMatrix4fv(ATTR_VERTPROJMAT, 1, GL.GL_TRUE, quad.vert_proj_mat)
		GL.glUniformMatrix4fv(ATTR_TEXPROJMAT, 1, GL.GL_TRUE, quad.tex_proj_mat)
		GL.glBindBuffer(GL.GL_ARRAY_BUFFER, quad.VBO)
		GL.glVertexAttribPointer(ATTR_POSITION, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
		GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(quad.vertices))
		GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
	GL.glBindVertexArray(0)

	sdl2.SDL_GL_SwapWindow(window.window)

class Quad:
	def __init__(self, ul, ur, br, bl, model_view_mat, texture):
		old = np.array([
			[-1.0,  1.0, 0.0],
			[ 1.0,  1.0, 0.0],
			[ 1.0, -1.0, 0.0],
			[-1.0, -1.0, 0.0]
			], dtype=np.float32)
		new = np.array([ul, ur, br, bl], dtype=np.float32)
		scale_bias = np.array([
			[0.5, 0.0, 0.0, 0.5],
			[0.0, 0.5, 0.0, 0.5],
			[0.0, 0.0, 1.0, 0.0],
			[0.0, 0.0, 0.0, 1.0]
			], dtype=np.float32)

		self.model_view_mat = model_view_mat
		self.vert_proj_mat = make_proj_mat(old, new)
		self.tex_proj_mat = np.dot(scale_bias, make_proj_mat(new, old))

		self.vertices = np.array([old[0], old[1], old[2], old[0], old[2], old[3]], dtype=np.float32).flatten()
		self.VBO = GL.glGenBuffers(1)
		GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.VBO)
		GL.glBufferData(GL.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL.GL_STATIC_DRAW)
		GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

		self.texture = texture


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

def run():
	global VAO_VERTEX
	global ATTR_POSITION
	global ATTR_TEXCOORDS
	global ATTR_TEXUNIT
	global ATTR_MODELVIEWMAT
	global ATTR_VERTPROJMAT
	global ATTR_TEXPROJMAT
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

	test_quads.append(Quad(
		[-1.0,  1.0, 0.0],
		[ 1.0,  0.3, 0.0],
		[ 1.0, -0.3, 0.0],
		[-1.0, -1.0, 0.0],
		make_model_view_mat(-0.6, 0.0, 0.0, 0.4, 0.4, 1.0), textures[1]))
	test_quads.append(Quad(
		[-1.0,  0.3, 0.0],
		[ 1.0,  1.0, 0.0],
		[ 1.0, -1.0, 0.0],
		[-1.0, -0.3, 0.0],
		make_model_view_mat(0.6, 0.0, 0.0, 0.4, 0.4, 1.0), textures[3]))

	while running:
		now = sdl2.timer.SDL_GetTicks()

		update()
		render()

		delay = 1000 / 60 - (sdl2.timer.SDL_GetTicks() - now)
		#print 1000 / (sdl2.timer.SDL_GetTicks() - now)
		if delay > 0:
			sdl2.timer.SDL_Delay(delay)

	window.hide()
	#sdl2.SDL_GL_DeleteContext(context)
	#sdl2.SDL_DestroyWindow(window.window)
	#sdl2.SDL_Quit()

run()