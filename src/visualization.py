import matplotlib.pyplot

def draw_occupancy_grid(ax, occupancy_grid):
	# Draw an occupancy grid
	pass

def draw_path(ax, path):
	# Draw the robot's path
	pass

def draw_pose_graph(ax, pose_graph, node_positions):
	# Draw the full pose graph (with all additional constraints)
	# node_positions should be an (n, 2) numpy array, encoding 
	# the (x, y) position of each of the nodes.
	
	# First, draw the edges
	for edge in pose_graph.graph.edges:
		ax.plot(node_positions[edge,0], node_positions[edge, 1], color="red")

	# Next, draw the nodes
	ax.scatter(node_positions[:,0], node_positions[:,1], color="red")

def draw_icp_iteration(ax, pc1, pc2, correspondences=None):
	# Draws the output of ICP at a single iteration, centered on
	# pc1, showing both point clouds, and optionally the correspondences
	# between them.
	pass