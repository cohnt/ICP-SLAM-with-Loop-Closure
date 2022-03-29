import numpy as np

import src.utils as utils

# NOTE: Our occupancy grid convention is that 0,0 is the bottom-left corner, and
# each successive row moves "up" in the world. When saving the occupancy grid as
# a map (for use with the robot), we flip it so that each row moves "down" in the
# world.

def produce_occupancy_grid(poses, lidar_points, cell_width, kHitOdds=3, kMissOdds=1):
	# Produce an occupancy grid from range data
	# poses should be an (n, 3) numpy array of poses (x, y, theta)
	# lidar_points should be an (n, m, 2) numpy array of point cloud
	#    lidar data -- each m-by-2 entry of the length-n list is in
	#    its own local coordinate frame (i.e. the robot center is at
	#    0,0, and the forward direction is positive x)
	# containing the complete map (with each cell being cell_width)
	# The occupancy grid will be a (h, w) numpy array of integers between
	# -128 and 127, with dtype np.int8

	n = poses.shape[0]
	m = lidar_points.shape[1]

	global_points = np.zeros(lidar_points.shape)
	for i in range(n):
		pose_tf = utils.odom_change_to_mat(poses[i])
		for j in range(m):
			point_homogeneous = np.array([[lidar_points[i,j,0]], [lidar_points[i,j,1]], [1]])
			global_points[i,j] = (pose_tf @ point_homogeneous).flatten()[:2]

	min_x = np.min(global_points[:,:,0]) - (cell_width / 2)
	max_x = np.max(global_points[:,:,0]) + (cell_width / 2)
	min_y = np.min(global_points[:,:,1]) - (cell_width / 2)
	max_y = np.max(global_points[:,:,1]) + (cell_width / 2)
	width_dist = max_x - min_x
	height_dist = max_y - min_y

	width_in_cells = np.ceil(width_dist / cell_width).astype(int)
	height_in_cells = np.ceil(height_dist / cell_width).astype(int)

	occupancy_grid = np.zeros((height_in_cells, width_in_cells), dtype=np.int8)

	for i in range(n):
		for j in range(m):
			y0, x0 = global_position_to_grid_cell(poses[i,:2], min_x, min_y, cell_width)
			y1, x1 = global_position_to_grid_cell(global_points[i,j], min_x, min_y, cell_width)
			dx = np.abs(x1 - x0).astype(int)
			dy = -np.abs(y1 - y0).astype(int)
			sx = 1 if x1 > x0 else -1
			sy = 1 if y1 > y0 else -1
			error = dx + dy

			while True:
				if x0 < 0 or x0 >= width_in_cells or y0 < 0 or y0 >= height_in_cells:
					break

				if -128 - occupancy_grid[y0, x0] < -kMissOdds:
					occupancy_grid[y0, x0] = occupancy_grid[y0, x0] - kMissOdds
				else:
					occupancy_grid[y0, x0] = -128

				e2 = error * 2
				if e2 >= dy:
					if x0 == x1:
						break
					error = error + dy
					x0 += sx
				if e2 <= dx:
					if y0 == y1:
						break
					error = error + dx
					y0 += sy

			if x0 >= 0 and x0 < width_in_cells and y0 >= 0 and y0 < height_in_cells:
				if 127 - occupancy_grid[y0, x0] > kHitOdds:
					occupancy_grid[y0, x0] = occupancy_grid[y0, x0] + kHitOdds
				else:
					occupancy_grid[y0, x0] = 127

	return occupancy_grid

def global_position_to_grid_cell(pos, min_x, min_y, cell_width):
	# Given an (x, y) position pos, as well as various information about
	# the occupancy grid, return the grid cell index (row, column)
	horizontal = np.floor((pos[0] - min_x) / cell_width).astype(int)
	vertical = np.floor((pos[1] - min_y) / cell_width).astype(int)
	return (vertical, horizontal)