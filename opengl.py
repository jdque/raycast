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
VAO = None
VBO = None
texture = None
running = True

ATTR_POSITION = -1
ATTR_TEXCOORDS = -1
ATTR_TEXUNIT = -1
ATTR_MODELVIEWMAT = -1
ATTR_PROJMAT = -1

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
		uniform mat4 v_projMat;
		noperspective out vec4 f_texCoords;

		void main()
		{
			f_texCoords = v_projMat * vec4(v_position.x, v_position.y, 0, 1);
			gl_Position = v_modelViewMat * vec4(v_position, 1);
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

	modelViewMat = np.array([
		[ 1.0, 0.0, 0.0, -1.0],
		[ 0.0, 1.0, 0.0,  0.0],
		[ 0.0, 0.0, 1.0,  0.0],
		[ 0.0, 0.0, 0.0,  1.0]
		])
	GL.glUniformMatrix4fv(ATTR_MODELVIEWMAT, 1, GL.GL_TRUE, modelViewMat)

	scale_bias = np.array([
		[0.5, 0.0, 0.0, 0.5],
		[0.0, 0.5, 0.0, 0.5],
		[0.0, 0.0, 1.0, 0.0],
		[0.0, 0.0, 0.0, 1.0]
		])
	proj_mat = np.dot(scale_bias, test_make_transform())
	GL.glUniformMatrix4fv(ATTR_PROJMAT, 1, GL.GL_TRUE, proj_mat)

	GL.glActiveTexture(GL.GL_TEXTURE0)
	GL.glBindTexture(GL.GL_TEXTURE_2D, texture)

	GL.glBindVertexArray(VAO)
	GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
	GL.glBindVertexArray(0)

	sdl2.SDL_GL_SwapWindow(window.window)

def test_make_transform():
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

	old = np.array([
		[-1.0,  1.0, 0.0],
		[ 1.0,  0.3, 0.0],
		[ 1.0, -0.3, 0.0],
		[-1.0, -1.0, 0.0]
		], dtype=np.float32)

	new = np.array([
		[-1.0,  1.0, 0.0],
		[ 1.0,  1.0, 0.0],
		[ 1.0, -1.0, 0.0],
		[-1.0, -1.0, 0.0]
		], dtype=np.float32)

	coeffs = []
	rhs = []
	for o, n in zip(old, new):
		coeffs.append(np.array([o[0], o[1], 1, 0, 0, 0, -o[0] * n[0], -o[1] * n[0]]))
		coeffs.append(np.array([0, 0, 0, o[0], o[1], 1, -o[0] * n[1], -o[1] * n[1]]))
		rhs.append(n[0])
		rhs.append(n[1])

	X = np.linalg.solve(np.array(coeffs), np.array(rhs))

	trans_mat = np.array([
		[X[0], X[1],  0., X[2]],
		[X[3], X[4],  0., X[5]],
		[  0.,   0.,  1.,   0.],
		[X[6], X[7],  0.,   1.]
		], dtype=np.float32)

	return trans_mat

def run():
	global VAO
	global VBO
	global ATTR_POSITION
	global ATTR_TEXCOORDS
	global ATTR_TEXUNIT
	global ATTR_MODELVIEWMAT
	global ATTR_PROJMAT
	global texture

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

	VAO = GL.glGenVertexArrays(1)
	GL.glBindVertexArray(VAO)

	VBO_positions = GL.glGenBuffers(1)
	GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO_positions)
	GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)

	VBO_tex_coords = GL.glGenBuffers(1)
	GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO_tex_coords)
	GL.glBufferData(GL.GL_ARRAY_BUFFER, tex_coords.nbytes, tex_coords, GL.GL_STATIC_DRAW)

	ATTR_POSITION = GL.glGetAttribLocation(shaderProgram, "v_position")
	GL.glEnableVertexAttribArray(ATTR_POSITION)
	GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO_positions)
	GL.glVertexAttribPointer(ATTR_POSITION, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

	# ATTR_TEXCOORDS = GL.glGetAttribLocation(shaderProgram, "v_texCoords")
	# GL.glEnableVertexAttribArray(ATTR_TEXCOORDS)
	# GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO_tex_coords)
	# GL.glVertexAttribPointer(ATTR_TEXCOORDS, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

	ATTR_TEXUNIT = GL.glGetUniformLocation(shaderProgram, "f_texUnit")
	GL.glProgramUniform1i(shaderProgram, ATTR_TEXUNIT, 0)

	ATTR_MODELVIEWMAT = GL.glGetUniformLocation(shaderProgram, "v_modelViewMat")

	ATTR_PROJMAT = GL.glGetUniformLocation(shaderProgram, "v_projMat")

	GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
	GL.glBindVertexArray(0)

	texture_surface = sdl2.ext.load_image("C:\\dev\\raycast\\textures.png")
	sprite_factory = sdl2.ext.SpriteFactory(sprite_type=sdl2.ext.SOFTWARE, renderer=renderer)
	test_sprite = sprite_factory.create_software_sprite((64, 64))
	sdl2.SDL_BlitSurface(texture_surface, sdl2.SDL_Rect(64, 0, 64, 64), test_sprite.surface, sdl2.SDL_Rect(0, 0, 64, 64))

	pixels = sdl2.ext.pixels2d(test_sprite)
	pixels = np.copy(pixels)
	texture = GL.glGenTextures(1)
	GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
	GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
	GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, 64, 64, 0, GL.GL_BGRA, GL.GL_UNSIGNED_INT_8_8_8_8_REV, np.flipud(pixels.T).reshape(64*64))
	GL.glGenerateMipmap(GL.GL_TEXTURE_2D)

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