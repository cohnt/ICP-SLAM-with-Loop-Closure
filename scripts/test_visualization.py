#######################
# Fix import issues   #
import sys            #
sys.path.append(".")  #
sys.path.append("..") #
#######################

import numpy as np
import matplotlib.pyplot as plt

import src.visualization as visualization
import src.pose_graph as pose_graph
import src.utils as utils

occupancy_grid = np.random.randint(-128, 128, size=(30, 30), dtype=np.int8)
odometry = np.array([
	[0, 0, 0],
	[1, 0, 0],
	[2, 0, 0],
	[2, 1, np.pi/2],
	[2, 2, np.pi/2]
])
path = odometry[:,:2]
g = pose_graph.PoseGraph(odometry)
g.add_constraint(0, 4, np.eye(3))

fig, ax = plt.subplots()
visualization.draw_occupancy_grid(ax, occupancy_grid, cell_size=0.1, origin_location=np.array([0, 0]))
plt.show()

fig, ax = plt.subplots()
visualization.draw_occupancy_grid(ax, occupancy_grid, cell_size=0.1, origin_location=np.array([0, 0]))
visualization.draw_path(ax, path)
plt.show()

fig, ax = plt.subplots()
visualization.draw_occupancy_grid(ax, occupancy_grid, cell_size=0.1, origin_location=np.array([0, 0]))
visualization.draw_path(ax, path)
visualization.draw_pose_graph(ax, g, path)
plt.show()

pc1 = np.random.random(size=(10,2))
pc2 = np.random.random(size=(10,2))
correspondences = np.array([0, 1, 2, 5], dtype=int)

fig, ax = plt.subplots()
visualization.draw_icp_iteration(ax, pc1, pc2)
plt.show()

fig, ax = plt.subplots()
visualization.draw_icp_iteration(ax, pc1, pc2, correspondences=correspondences)
plt.show()