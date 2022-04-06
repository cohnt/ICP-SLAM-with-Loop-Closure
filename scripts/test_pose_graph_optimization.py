#######################
# Fix import issues   #
import sys            #
sys.path.append(".")  #
sys.path.append("..") #
#######################

import numpy as np
import matplotlib.pyplot as plt

import src.dataloader as dataloader
import src.icp as icp
import src.loop_closure_detection as loop_closure_detection
import src.manual_loop_closure as manual_loop_closure
import src.pose_graph as pose_graph
import src.pose_graph_optimization as pose_graph_optimization
import src.produce_occupancy_grid as produce_occupancy_grid
import src.store_pose_graph as store_pose_graph
import src.utils as utils
import src.visualization as visualization

side_len = 3.0 # Meters
poses_per_side = 30
num_loops = 4
sides_in_a_square = 4

pos_noise_var = 0.01
theta_noise_var = 0.025

num_constriants = 100

poses = [] # Will be an n-by-3 list of poses, each pose is (x, y, theta)
current_pose = [0, 0, 0]

for _ in range(num_loops):
	for _ in range(4):
		for _ in range(poses_per_side):
			poses.append(current_pose.copy())
			current_pose[0] += (side_len / poses_per_side) * np.cos(current_pose[2])
			current_pose[1] += (side_len / poses_per_side) * np.sin(current_pose[2])
			current_pose[0] += np.random.normal(loc=0, scale=pos_noise_var)
			current_pose[1] += np.random.normal(loc=0, scale=pos_noise_var)
			current_pose[2] += np.random.normal(loc=0, scale=theta_noise_var) % (2 * np.pi)
		current_pose[2] += np.pi / 2
poses = np.array(poses)

plt.scatter(poses[:,0], poses[:,1])
plt.show()

pg = pose_graph.PoseGraph(poses)

constraint_idx = np.random.choice(poses_per_side * sides_in_a_square, num_constriants, replace=True)
loop_count = [np.random.choice(num_loops, 2, replace=False) for _ in constraint_idx]

for i in range(len(constraint_idx)):
	a = constraint_idx[i] + (poses_per_side * sides_in_a_square * loop_count[i][0])
	b = constraint_idx[i] + (poses_per_side * sides_in_a_square * loop_count[i][1])
	pg.add_constraint(a, b, np.eye(3))

pg.add_constraint(0, poses_per_side * sides_in_a_square, np.eye(3))
pg.add_constraint(len(poses)-1, len(poses)-1 - (poses_per_side * sides_in_a_square), np.eye(3))

fig, ax = plt.subplots()
visualization.draw_pose_graph(ax, pg, draw_nodes=True, draw_orientation=True)
visualization.draw_path(ax, poses[:,:2])
plt.draw()
plt.pause(0.1)

iters = 0
while True:
	iters += 1
	if iters % 5 == 0:
		pg.flip()
	pose_graph_optimization.pose_graph_optimization_step(pg)
	ax.cla()
	visualization.draw_pose_graph(ax, pg, draw_nodes=True, draw_orientation=True)
	visualization.draw_path(ax, pg.poses[:,:2])
	plt.draw()
	plt.pause(0.1)