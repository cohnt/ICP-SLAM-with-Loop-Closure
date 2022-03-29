import networkx as nx

# Pose graphs are directed networkx graphs. Nodes are labeled with numerical IDs, matching their
# index in the original data. Edges are 3x3 numpy matrices in SE(2), which denote the transformation
# between the nodes. In the following case:
#
# (1) -> (2)
#     T
#
# T is the transformation from (1) to (2)

class pose_graph():
	def __init__(self, transformations):
		# transformations should be an (n-1, 3, 3) numpy array, where the ith entry is the SE(3) matrix
		# of the transformation from pose i to pose i+1. Returns a pose graph.
		pass

	def add_constraint(self, i, j, transformation):
		# Adds the transformation from i to j into the graph object
		# TODO: Make sure this is by reference!
		pass