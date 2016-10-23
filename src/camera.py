import numpy as np

class Camera:
	def __init__(self):
		self.pos = None
		self.z = 0.
		self.angle = 0.
		self.near = 0.
		self.far = 0.
		self.proj_width = 0.
		self.proj_height = 0.
		self.aspect = 0.
		self.horizon_y = 0.
		self.near_dir = None
		self.near_plane = None
		self.rays = None

		self.pos = np.array([0., 0.])
		self.z = 0.
		self.set_fov(90, 0, 100, 100, 100)

	def move_to(self, x, y, z):
		self.pos[0] = x
		self.pos[1] = y
		self.z = z

	def move_by(self, dx, dy, dz):
		self.pos[0] += dx
		self.pos[1] += dy
		self.z += dz

	def move_forward(self, distance):
		step = self.near_dir / self.near * distance
		self.move_by(step[0], step[1], 0)

	def rotate_to(self, angle):
		rad = np.deg2rad(angle)
		rot = np.array([
			[np.cos(rad), -np.sin(rad)],
			[np.sin(rad), np.cos(rad)]
		])
		#TODO - this is wrong
		self.near_dir = np.dot(rot, self.near_dir)
		self.near_plane = np.dot(rot, self.near_plane)

	def rotate_by(self, da):
		rad = np.deg2rad(da)
		rot = np.array([
			[np.cos(rad), -np.sin(rad)],
			[np.sin(rad), np.cos(rad)]
		])
		self.near_dir = np.dot(rot, self.near_dir)
		self.near_plane = np.dot(rot, self.near_plane)

		for i in xrange(0, len(self.rays)):
			self.rays[i] = np.dot(rot, self.rays[i])

	def tilt_by(self, distance):
		self.horizon_y += distance

	def set_fov(self, angle, near, far, proj_width, proj_height):
		self.angle = float(angle)
		self.near = float(near)
		self.far = float(far)
		self.proj_width = float(proj_width)
		self.proj_height = float(proj_height)
		self.aspect = float(self.proj_width / self.proj_height)
		self.horizon_y = float(proj_height) / 2

		self.near_dir = np.array([self.near, 0])
		self.near_plane = np.array([0, self.near * np.tan(np.deg2rad(self.angle / 2))])
		self.rays = self.generate_rays()

	def generate_rays(self):
		rays = []
		dir_vec = np.array([(self.proj_width / 2) / np.tan(np.deg2rad(self.angle / 2)), 0])
		unit_plane = np.array([0, 1])
		for i in xrange(int(-self.proj_width / 2), int(self.proj_width / 2)):
			plane_pt = dir_vec + (unit_plane * i)
			unit_ray = plane_pt / np.linalg.norm(plane_pt)
			rays.append(unit_ray)
		return np.array(rays)