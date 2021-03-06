import os
os.environ["PYSDL2_DLL_PATH"] = os.path.dirname(__file__) + os.path.sep + ".." + os.path.sep + "lib"

import sdl2
import sdl2.ext
import sdl2.video

import OpenGL.GL as GL
from OpenGL.GL import shaders

import numpy as np
from engine import get_clipped_tile_points, get_tri_quads

window = None
surface = None
renderer = None
context = None
shaderProgram = None
textures = []
running = False

test_tris = []

VAO_VERTEX = -1
ATTR_POSITION = -1
ATTR_TEXUNIT = -1
ATTR_MODELVIEWMAT = -1
ATTR_VERTPROJMAT = -1
ATTR_TEXPROJMAT = -1

VBO = None

def render_raycast(camera, tilemap):
    global test_tris

    test_tris = []
    floor_pts, wall_pts = get_clipped_tile_points(tilemap, camera)
    quads = get_tri_quads(floor_pts + wall_pts, camera)

    for quad in quads:
        o_tri = quad[0]
        t_tri = quad[1]
        pos = quad[2]
        offset = quad[3]
        texture = quad[4]
        surface_type = quad[5]

        o_tri[:,1] *= -1
        t_tri[:,1] *= -1
        pos[1] *= -1

        area = np.linalg.det(t_tri[0:3])
        if abs(area) <= 1e-10:
            continue

        test_tris.append(TriQuad(
            o_tri,
            t_tri,
            make_model_view_mat(pos[0], pos[1], 0.0, 1.0, 1.0, 1.0),
            offset,
            texture))

    vertices = np.zeros(len(test_tris) * 9, dtype=np.float32)
    for i in xrange(0, len(test_tris)):
        vertices[9*i:9*i+9] = test_tris[i].vertices

    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_DYNAMIC_DRAW)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

def update_raycast(camera, tilemap):
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
    if key_states[sdl2.SDL_SCANCODE_A]:
        camera.tilt_by(-5)
    elif key_states[sdl2.SDL_SCANCODE_D]:
        camera.tilt_by(5)

#------------------------------------------------------------------------------------------

class TriQuad:
    def __init__(self, original, transformed, model_view_mat, texture_bounds, texture_id):
        self.model_view_mat = model_view_mat
        self.vert_proj_mat = make_proj_mat(original, transformed)
        self.tex_proj_mat = make_proj_mat(transformed, texture_bounds)
        original[:,2] = transformed[:,2]
        self.vertices = original[0:3].ravel().astype(np.float32)
        self.texture_id = texture_id

def make_model_view_mat(x, y, z, sx, sy, sz):
    model_view_mat = np.array([
        [ sx, 0.0, 0.0,  x ],
        [0.0,  sy, 0.0,  y ],
        [0.0, 0.0,  sz,  z ],
        [0.0, 0.0, 0.0, 1.0]
        ])
    return model_view_mat

def make_proj_mat(orig_pts, trans_pts):
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

    coeffs = np.empty((len(orig_pts) * 2, 8))
    rhs = np.empty((len(orig_pts) * 2))
    for i in xrange(len(orig_pts)):
        o = orig_pts[i]
        t = trans_pts[i]
        coeffs[i*2] = [o[0], o[1], 1, 0, 0, 0, -o[0] * t[0], -o[1] * t[0]]
        coeffs[i*2+1] = [0, 0, 0, o[0], o[1], 1, -o[0] * t[1], -o[1] * t[1]]
        rhs[i*2] = t[0]
        rhs[i*2+1] = t[1]

    X = np.linalg.solve(coeffs, rhs)

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
    global VAO_VERTEX
    global ATTR_POSITION
    global ATTR_TEXUNIT
    global ATTR_MODELVIEWMAT
    global ATTR_VERTPROJMAT
    global ATTR_TEXPROJMAT
    global VBO
    global textures

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

        uniform mat4 v_modelViewMat;
        uniform mat4 v_vertProjMat;
        uniform mat4 v_texProjMat;

        noperspective out vec4 f_texCoords;
        out float f_z;

        void main()
        {
            vec4 projPos = v_vertProjMat * vec4(v_position, 1);
            f_texCoords = v_texProjMat * (projPos / projPos.w);
            f_z = projPos.z;
            gl_Position = v_modelViewMat * projPos;
        }
        """, GL.GL_VERTEX_SHADER)

    fragmentShader = GL.shaders.compileShader("""
        #version 430

        uniform sampler2D f_texUnit;

        noperspective in vec4 f_texCoords;
        in float f_z;

        void main()
        {
            //gl_FragColor = vec4(color, 1.0);
            //gl_FragColor = texture(f_texUnit, f_texCoords);
            //vec2 newCoords = vec2(f_texCoords.x / f_texCoords.w, f_texCoords.y / f_texCoords.w);
            //gl_FragColor = vec4( vec2(f_texCoords.x / f_texCoords.w, f_texCoords.y / f_texCoords.w),0,1);
            float z = (1.0 - f_z - 0.6) * 4.0;
            //gl_FragColor = vec4(z, z, z, 1.0);
            gl_FragDepth = f_z;
            gl_FragColor = texture2DProj(f_texUnit, f_texCoords) * vec4(z, z, z, 1.0);
        }
        """, GL.GL_FRAGMENT_SHADER)

    shaderProgram = GL.shaders.compileProgram(vertexShader, fragmentShader)

    #Shader attributes
    ATTR_MODELVIEWMAT = GL.glGetUniformLocation(shaderProgram, "v_modelViewMat")
    ATTR_VERTPROJMAT = GL.glGetUniformLocation(shaderProgram, "v_vertProjMat")
    ATTR_TEXPROJMAT = GL.glGetUniformLocation(shaderProgram, "v_texProjMat")
    ATTR_TEXUNIT = GL.glGetUniformLocation(shaderProgram, "f_texUnit")

    VAO_VERTEX = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(VAO_VERTEX)
    ATTR_POSITION = GL.glGetAttribLocation(shaderProgram, "v_position")
    GL.glEnableVertexAttribArray(ATTR_POSITION)
    GL.glBindVertexArray(0)

    VBO = GL.glGenBuffers(1)

    #Load textures
    texture_surface = sdl2.ext.load_image(os.path.dirname(__file__) + os.path.sep + ".." + os.path.sep + "assets/textures.png")
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
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT);
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT);
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        textures.append(texture)

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
    #GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)

    GL.glActiveTexture(GL.GL_TEXTURE0)

    GL.glBindVertexArray(VAO_VERTEX)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
    GL.glVertexAttribPointer(ATTR_POSITION, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
    for i in xrange(0, len(test_tris)):
        tri = test_tris[i]
        GL.glUniformMatrix4fv(ATTR_MODELVIEWMAT, 1, GL.GL_TRUE, tri.model_view_mat)
        GL.glUniformMatrix4fv(ATTR_VERTPROJMAT, 1, GL.GL_TRUE, tri.vert_proj_mat)
        GL.glUniformMatrix4fv(ATTR_TEXPROJMAT, 1, GL.GL_TRUE, tri.tex_proj_mat)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tri.texture_id)
        GL.glDrawArrays(GL.GL_TRIANGLES, i * 3, 3)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
    GL.glBindVertexArray(0)

    sdl2.SDL_GL_SwapWindow(window.window)

def run_test():
    global running

    init()
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

    running = True
    while running:
        update()
        render()
    window.hide()

def run(tilemap, camera):
    global running

    init()

    running = True
    while running:
        now = sdl2.timer.SDL_GetTicks()

        update()
        update_raycast(camera, tilemap)
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