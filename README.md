# Raycast

Experimental 3D rendering project that combines old-school [raycasting](http://permadi.com/1996/05/ray-casting-tutorial-table-of-contents/) with GPU texture mapping.

#### Raycasting

This project follows the raycasting process of projecting a 2D scene onto screen space. For each pixel column on the screen, a ray is cast outwards from the person's position on the 2D map. The distance the ray travels to the first wall it intersects determines the height at which to render that columnar slice on the screen (closer walls will appear taller, and vice versa). This method is simple and works great for rendering walls, but floors and ceilings end up requiring pixel-by-pixel rendering. Performance quickly suffers as a result, especially as screen resolution increases.

#### Solution

The goal was to keep using raycasting to project geometry onto screen space, but also use OpenGL to paint textures on the walls, floors, and ceilings. This is done with the following procedure:

1. Triangulate the geometry in screen space
2. Reverse transform the geometry from screen coordinates to normalized device coordinates
3. For each triangle, construct its perspective projection matrix [1]
4. Send the triangle vertices and projection matrix to GLSL shaders
5. The corresponding texture for the triangle is then drawn in a perspective-correct manner

Most of the geometric transformations are vectorized via NumPy to bring it up to acceptable performance. There is still much room for improvement, such as in the collision detection and shader matrix binding.

---

[1] The following linear system is solved to construct a perspective projection matrix given four points on a plane:

```
|s|   |m1, m2,  0, m3|   |x|
|t| = |m4, m5,  0, m6| * |y|
|0|   | 0,  0,  1,  0|   |0|
|w|   |m7, m8,  0,  1|   |1|

x, y  = transformed corner vertices in object coordinates
s, t  = original corner vertices in object coordinates
s/w, t/w = projected texture coordinates
```