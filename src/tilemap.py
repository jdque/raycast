from collections import namedtuple
import numpy as np
from geometry import round_nearest, round_down

LEFT = 0
TOP = 1
RIGHT = 2
BOTTOM = 3

SimpleTile = namedtuple('SimpleTile', [
    'floor_z',
    'floor_height',
    'floor_texture',
    'wall_texture'
    ])

Floor = namedtuple('Floor', [
    'z',
    'height',
    'texture'
    ])

Wall = namedtuple('Wall', [
    'z',
    'height',
    'texture'
    ])

class Palette:
    def __init__(self):
        self.palette = {}

    def add(self, id, tile):
        self.palette[id] = tile

    def get(self, id):
        return self.palette[id]

class TileMap:
    def __init__(self, width, height, size):
        self.width = width
        self.height = height
        self.size = size
        self.floors = np.empty([self.height, self.width], dtype=object)
        self.wall_groups = np.empty([self.height, self.width], dtype=tuple)

    def set_floors_from_palette(self, palette, floors):
        if len(floors[0]) != self.width or len(floors) != self.height:
            return
        for y in xrange(0, self.height):
            for x in xrange(0, self.width):
                self.floors[y,x] = palette.get(floors[y][x])

    def set_wall_groups_from_palette(self, palette, wall_groups):
        if len(wall_groups[0]) != self.width or len(wall_groups) != self.height:
            return
        for y in xrange(0, self.height):
            for x in xrange(0, self.width):
                self.wall_groups[y,x] = palette.get(wall_groups[y][x])

    def set_tiles_from_palette(self, palette, tiles):
        if len(tiles[0]) != self.width or len(tiles) != self.height:
            return
        for y in xrange(0, self.height):
            for x in xrange(0, self.width):
                tile = palette.get(tiles[y][x])
                self.floors[y,x] = Floor(tile.floor_z, tile.floor_height, tile.floor_texture)
                self.wall_groups[y,x] = [Wall(tile.floor_z, tile.floor_height, tile.wall_texture)] * 4

    def get_floor(self, x, y):
        return self.floors[y,x]

    def get_floor_px(self, px, py):
        x = px / self.size
        y = py / self.size
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return None
        return self.floors[y,x]

    def get_walls_px(self, px, py):
        tile_size = self.size
        is_vert_wall = abs(px % tile_size - tile_size) <= 1 or abs(px % tile_size) <= 1
        is_horz_wall = abs(py % tile_size - tile_size) <= 1 or abs(py % tile_size) <= 1
        if not is_horz_wall and not is_vert_wall:
            return [], []

        walls = []
        tile_refs = []

        if is_horz_wall:
            x = round_down(px, tile_size) / int(tile_size)
            y = round_nearest(py, tile_size) / int(tile_size)
            if x < self.width:
                if y == 0:
                    walls += [self.wall_groups[y,x][TOP]]
                    tile_refs += [(x, y)]
                elif y == self.height:
                    walls += [self.wall_groups[y-1,x][BOTTOM]]
                    tile_refs += [(x, y-1)]
                else:
                    walls += [self.wall_groups[y-1,x][BOTTOM], self.wall_groups[y,x][TOP]]
                    tile_refs += [(x, y-1), (x, y)]
        if is_vert_wall:
            x = round_nearest(px, tile_size) / int(tile_size)
            y = round_down(py, tile_size) / int(tile_size)
            if y < self.height:
                if x == 0:
                    walls += [self.wall_groups[y,x][LEFT]]
                    tile_refs += [(x, y)]
                elif x == self.width:
                    walls += [self.wall_groups[y,x-1][RIGHT]]
                    tile_refs += [(x-1, y)]
                else:
                    walls += [self.wall_groups[y,x-1][RIGHT], self.wall_groups[y,x][LEFT]]
                    tile_refs += [(x-1, y), (x, y)]

        return walls, tile_refs