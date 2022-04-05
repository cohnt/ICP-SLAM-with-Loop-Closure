import numpy as np

import src.utils as utils

# NOTE: Our occupancy grid convention is that 0,0 is the bottom-left corner, and
# each successive row moves "up" in the world. When saving the occupancy grid as
# a map (for use with the robot), we flip it so that each row moves "down" in the
# world.

def produce_occupancy_grid(poses, lidar_points, cell_width, min_width=0, min_height=0, kHitOdds=3, kMissOdds=1):
	# Produce an occupancy grid from range data
	# poses should be an (n, 3) numpy array of poses (x, y, theta)
	# lidar_points should be an length-n array of (m_i, 2) numpy arrays of point cloud
	#    lidar data -- each m_i-by-2 entry of the length-n list is in
	#    its own local coordinate frame (i.e. the robot center is at
	#    0,0, and the forward direction is positive x)
	# min_width and min_height are the minimum map sizes, in meters
	# containing the complete map (with each cell being cell_width)
	# The occupancy grid will be a (h, w) numpy array of integers between
	# -128 and 127, with dtype np.int8
	# Also returns the minimum x and y values of the grid

	n = poses.shape[0]
	ms = [len(lidar_points[i]) for i in range(len(lidar_points))]

	global_points = construct_global_points(poses, lidar_points)

	all_points = np.concatenate(global_points)
	min_x = np.min(all_points[:,0]) - (cell_width / 2)
	max_x = np.max(all_points[:,0]) + (cell_width / 2)
	min_y = np.min(all_points[:,1]) - (cell_width / 2)
	max_y = np.max(all_points[:,1]) + (cell_width / 2)
	width_dist = max_x - min_x
	height_dist = max_y - min_y

	if width_dist < min_width:
		offset = (min_width - width_dist) / 2
		min_x -= offset
		max_x += offset
		width_dist = min_width
	if height_dist < min_height:
		offset = (min_height - height_dist) / 2
		min_y -= offset
		max_y += offset
		height_dist = min_height

	width_in_cells = np.ceil(width_dist / cell_width).astype(int)
	height_in_cells = np.ceil(height_dist / cell_width).astype(int)

	occupancy_grid = np.zeros((height_in_cells, width_in_cells), dtype=np.int8)

	for i in range(n):
		print("Processing frame %d" % i)
		for j in range(ms[i]):
			bresenham_update(occupancy_grid, poses[i,:2], global_points[i][j], min_x, min_y, cell_width, kHitOdds, kMissOdds)

	return occupancy_grid, (min_x, min_y)

def update_occupancy_grid(occupancy_grid, poses, lidar_points, cell_width, min_x, min_y, kHitOdds=3, kMissOdds=1):
	# Update an existing occupancy grid from new range data
	# poses should be an (n, 3) numpy array of poses (x, y, theta)
	# lidar_points should be an length-n array of (m_i, 2) numpy arrays of point cloud
	#    lidar data -- each m_i-by-2 entry of the length-n list is in
	#    its own local coordinate frame (i.e. the robot center is at
	#    0,0, and the forward direction is positive x)
	# containing the complete map (with each cell being cell_width)
	# The occupancy grid will be a (h, w) numpy array of integers between
	# -128 and 127, with dtype np.int8
	n = poses.shape[0]
	ms = [len(lidar_points[i]) for i in range(len(lidar_points))]

	global_points = construct_global_points(poses, lidar_points)

	for i in range(n):
		for j in range(ms[i]):
			bresenham_update(occupancy_grid, poses[i,:2], global_points[i][j], min_x, min_y, cell_width, kHitOdds, kMissOdds)

	return occupancy_grid

def construct_global_points(poses, lidar_points):
	# Transforms the points in lidar_points into the global frame, and
	# concatenates them into a single list
	n = poses.shape[0]
	ms = [len(lidar_points[i]) for i in range(len(lidar_points))]

	global_points = []
	for i in range(n):
		pose_tf = utils.odom_change_to_mat(poses[i])
		global_points.append(np.zeros(lidar_points[i].shape))
		for j in range(ms[i]):
			point_homogeneous = np.array([[lidar_points[i][j,0]], [lidar_points[i][j,1]], [1]])
			global_points[i][j] = (pose_tf @ point_homogeneous).flatten()[:2]
	return global_points

def bresenham_update(occupancy_grid, pose, point, min_x, min_y, cell_width, kHitOdds, kMissOdds):
	# Update the given occupancy grid for a given lidar beam, based on the various input parameters
	y0, x0 = global_position_to_grid_cell(pose, min_x, min_y, cell_width)
	y1, x1 = global_position_to_grid_cell(point, min_x, min_y, cell_width)
	dx = np.abs(x1 - x0).astype(int)
	dy = -np.abs(y1 - y0).astype(int)
	sx = 1 if x1 > x0 else -1
	sy = 1 if y1 > y0 else -1
	error = dx + dy

	while True:
		if x0 < 0 or x0 >= occupancy_grid.shape[1] or y0 < 0 or y0 >= occupancy_grid.shape[0]:
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

	if x0 >= 0 and x0 < occupancy_grid.shape[1] and y0 >= 0 and y0 < occupancy_grid.shape[0]:
		if 127 - occupancy_grid[y0, x0] > kHitOdds:
			occupancy_grid[y0, x0] = occupancy_grid[y0, x0] + kHitOdds
		else:
			occupancy_grid[y0, x0] = 127

def global_position_to_grid_cell(pos, min_x, min_y, cell_width):
	# Given an (x, y) position pos, as well as various information about
	# the occupancy grid, return the grid cell index (row, column)
	horizontal = np.floor((pos[0] - min_x) / cell_width).astype(int)
	vertical = np.floor((pos[1] - min_y) / cell_width).astype(int)
	return (vertical, horizontal)

def grid_mle(grid, unknown_empty=True):
	# Rounds every grid cell to either 127 or -128. If unknown_empty is True,
	# unkown points (i.e. log-likelihood 0) are set to -128 (empty). Otherwise,
	# they are set to 127 (occupied).
	grid = grid.copy()
	grid[grid > 0] = 127
	grid[grid < 0] = -128
	grid[grid == 0] = -128 if unknown_empty else 127
	return grid

def save_grid(grid, fname, cell_width):
	# Saves the grid into a map file, following the EECS 467 convention
	f = open(fname, "w")
	f.write("%d %d %d %d %f\n" % (0, 0, grid.shape[1], grid.shape[0], cell_width))
	for i in range(grid.shape[0]):
		for j in range(grid.shape[1]):
			f.write("%d " % grid[i][j])
		f.write("\n")
	f.close()