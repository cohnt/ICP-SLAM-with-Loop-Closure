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

odometry = np.array([
	[0, 0, 0],
	[1, 0, 0],
	[2, 0, 0],
	[2, 1, np.pi/2],
	[2, 2, np.pi/2]
])
path = odometry[:,:2]
pg = pose_graph.PoseGraph(odometry)
pg.add_constraint(0, 4, np.eye(3))

fig, ax = plt.subplots()
visualization.draw_path(ax, pg.poses[:,:2])
visualization.draw_pose_graph(ax, pg)
plt.show()

pg.save("temp.pickle")

new_pg = pose_graph.PoseGraph(None)
new_pg.load("temp.pickle")
pg = new_pg

fig, ax = plt.subplots()
visualization.draw_path(ax, pg.poses[:,:2])
visualization.draw_pose_graph(ax, pg)
plt.show()