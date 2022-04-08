import networkx as nx
import src.utils as utils
import numpy as np
import pickle

# Pose graphs are directed networkx graphs. Nodes are labeled with numerical IDs, matching their
# index in the original data. Edges are 3x3 numpy matrices in SE(2), which denote the transformation
# between the nodes. In the following case:
#
# (1) -> (2)
#     T
#
# T is the transformation from (1) to (2)

# To iterate over the constraints in a pose graph, use something like
# for edge in pose_graph.graph.edges.data('object'):
#      do stuff...
# Each "edge" will be a 3-tuple (i, j, transformation) where transfomation is the 3x3 matrix transformation
# from node i to node j.

class PoseGraph():
	def __init__(self, poses):
		# poses should be an (n,3) numpy array of poses, where the ith entry is an (x, y, theta) pose.
		# Returns a pose graph.
		# Call with poses == None only if you're going to load from a file
		self.poses = poses
		self.graph = nx.DiGraph()

		if poses is None:
			return

		successive_offset = poses[1:] - poses[:-1]
		successive_tf = [utils.odom_change_to_mat(offset) for offset in successive_offset]

		for i in range(0, len(poses)-1):
			self.graph.add_edge(i, i+1, object=successive_tf[i])

	def add_constraint(self, i, j, transformation):
		# Adds the transformation from i to j into the graph object
		self.graph.add_edge(i, j, object=transformation)

	def flip(self):
		# Switches the order of all edges, transformations, labels, etc.
		# Allows the pose graph optimization to work in both directions
		self.poses = self.poses[::-1]
		self.poses[:,2] = (self.poses[:,2] + np.pi) % (2 * np.pi)
		new_graph = nx.DiGraph()
		n = len(self.poses)-1
		for a, b, tf in self.graph.edges(data="object"):
			new_graph.add_edge(n-b, n-a, object=tf)
		self.graph = new_graph

	def save(self, fname):
		with open(fname, "wb") as f:
			pickle.dump((self.poses, self.graph), f)

	def load(self, fname):
		with open(fname, "rb") as f:
			self.poses, self.graph = pickle.load(f)