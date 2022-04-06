import numpy as np
import matplotlib.pyplot as plt

def draw_occupancy_grid(ax, occupancy_grid, cell_size, origin_location):
	# Draw an occupancy grid
	# occupancy_grid should be a (h, w) numpy array of integers between -128 and 127, with dtype np.int8
	# cell_size is the width of the occupancy grid cells (in meters)
	# origin_location is the (x, y) coordinates of the center of the bottom-left cell in occupancy_grid
	# Useful reference: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.imshow.html
	# Will need to use the "extent" argument
	height = occupancy_grid.shape[0] * cell_size
	width = occupancy_grid.shape[1] * cell_size
	half_cell_size = cell_size / 2.0
	left = origin_location[0] - half_cell_size
	right = origin_location[0] + width - half_cell_size
	bottom = origin_location[1] - half_cell_size
	top = origin_location[1] + height - half_cell_size
	ax.imshow(occupancy_grid, origin="lower", extent=(left, right, bottom, top), cmap=plt.get_cmap("gist_yarg"), vmin=-128, vmax=127)

def draw_path(ax, path, color="blue"):
	# Draw the robot's path
	# path should be an (n, 2) numpy array, encoding the (x, y) robot positions
	ax.plot(path[:,0], path[:,1], color=color)

def draw_pose_graph(ax, pose_graph, draw_nodes=False, node_size=0.05**2, draw_orientation=False, orientation_size=0.1, edge_color="red", node_color="blue", orien_color="green"):
	# Draw the full pose graph (with all additional constraints)

	# First, draw the edges
	for edge in pose_graph.graph.edges:
		ax.plot(pose_graph.poses[edge,0], pose_graph.poses[edge, 1], color=edge_color)

	# Next, draw the nodes
	if draw_nodes:
		ax.scatter(pose_graph.poses[:,0], pose_graph.poses[:,1], color=node_color, s=node_size)

	# Finally, draw the orientation
	if draw_orientation:
		for pose in pose_graph.poses:
			theta = pose[2]
			vec = np.array([np.cos(theta), np.sin(theta)]) * orientation_size
			xs = [pose[0], pose[0]+vec[0]]
			ys = [pose[1], pose[1]+vec[1]]
			ax.plot(xs, ys, color=orien_color)

def draw_icp_iteration(ax, pc1, pc2, correspondences=np.array([])):
	# Draws the output of ICP at a single iteration, centered on
	# pc1, showing both point clouds, and optionally the correspondences
	# between them. If it's given, correspondences should be an (n, 1)
	# numpy array of integers, showing the matched pairs. The first
	# entry in each row is the index of the point in pc1, and the second
	# entry in each row is the index of the point in pc2.

	# First, draw the matches (if specified)
	for i in range(correspondences.shape[0]):
		j = correspondences[i]
		xs = [pc1[i,0], pc2[j,0]]
		ys = [pc1[i,1], pc2[j,1]]
		ax.plot(xs, ys, color="black")

	# Then, draw the two point clouds
	ax.scatter(pc1[:,0], pc1[:,1], color="red")
	ax.scatter(pc2[:,0], pc2[:,1], color="blue")