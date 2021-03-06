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
import src.pose_graph as pose_graph
import src.pose_graph_optimization as pose_graph_optimization
import src.produce_occupancy_grid as produce_occupancy_grid
import src.utils as utils
import src.visualization as visualization

# This is the script that calls everything else. It will take in various command line arguments (such as filename, parameters), and runs SLAM

odometry, lidar_points = dataloader.parse_lcm_log("./data/EECS_3", load_images=False)
# odometry, lidar_points = dataloader.parse_lcm_log("./data/lab_maze", load_images=False)

# print(len(odometry))

dataloader.create_results_file_structure()

start = 75 # EECS
# start = 25 # LAB
cell_width = 0.05
# og, (min_x, min_y) = produce_occupancy_grid.produce_occupancy_grid(odometry[:start], lidar_points[:start], cell_width, min_width=3, min_height=3)

corrected_poses = np.array([odometry[0]])
for i in range(1, start):
	corrected_poses = np.vstack((corrected_poses, odometry[i]))

fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
# visualization.draw_occupancy_grid(ax, og, cell_size=cell_width, origin_location=np.array([min_x, min_y]))
# plt.draw()
# plt.pause(0.001)


skip = 5
assert skip < start
iters = 0
draw_every = 10
for i in range(start, len(odometry), skip):
	iters += 1
	estimated_tf = utils.pose_to_mat(odometry[i] - odometry[i-skip])
	pc_prev = np.c_[lidar_points[i-skip], np.ones(len(lidar_points[i-skip]))]
	pc_current = np.c_[lidar_points[i], np.ones(len(lidar_points[i]))]

	tfs, error = icp.icp(pc_current, pc_prev, init_transform=estimated_tf, max_iters=100, epsilon=0.05)

	# for j in range(0, len(tfs), 10):
	# 	pc = (tfs[j] @ pc_prev.T).T
	# 	plt.scatter(pc[:,0], pc[:,1], color="red")
	# 	plt.scatter(pc_current[:,0], pc_current[:,1], color="blue")
	# 	plt.show()

	# if len(corrected_poses) >= 48:
	# 	for tf in tfs:
	# 		pc = (tf @ pc_current.T).T
	# 		plt.scatter(pc[:,0], pc[:,1], color="red")
	# 		plt.scatter(pc_prev[:,0], pc_prev[:,1], color="blue")
	# 		plt.show()

	# pc = (tfs[0] @ pc_current.T).T
	# plt.scatter(pc[:,0], pc[:,1], color="red")
	# plt.scatter(pc_prev[:,0], pc_prev[:,1], color="blue")
	# plt.show()
	# pc = (tfs[-1] @ pc_current.T).T
	# plt.scatter(pc[:,0], pc[:,1], color="red")
	# plt.scatter(pc_prev[:,0], pc_prev[:,1], color="blue")
	# plt.show()

	# if error > 5:
	# 	real_tf = estimated_tf
	# else:
	# 	real_tf = tfs[-1]
	real_tf = tfs[-1]
	real_prev_pose = utils.pose_to_mat(corrected_poses[-1])
	real_pose = real_prev_pose @ real_tf
	real_odom = utils.mat_to_pose(real_pose)
	corrected_poses = np.vstack((corrected_poses, real_odom))

	pc1 = (utils.pose_to_mat(corrected_poses[-2]) @ pc_prev.T).T
	pc2 = (utils.pose_to_mat(corrected_poses[-1]) @ pc_current.T).T
	ax.scatter(pc1[::10,0], pc1[::10,1], color="red", s=0.1)
	ax.scatter(pc2[::10,0], pc2[::10,1], color="blue", s=0.1)
	
	visualization.draw_path(ax, corrected_poses)
	ax.set_aspect("equal")
	
	# plt.draw()
	# plt.pause(0.1)
	plt.savefig("results/iter%04d.png" % iters)
	print(iters)

	if(iters >= 200): # Use 200 for EECS_3 for now
		break

plt.close(fig)

print("Recorded %d poses. Creating occupancy grid..." % len(corrected_poses))
og, (min_x, min_y) = produce_occupancy_grid.produce_occupancy_grid(corrected_poses, lidar_points[:len(corrected_poses)], cell_width, kHitOdds=20, kMissOdds=10)
print("Drawing occupancy grid...")
fig, ax = plt.subplots()
visualization.draw_occupancy_grid(ax, og, cell_size=cell_width, origin_location=np.array([min_x, min_y]))
plt.savefig("results/final_map.png")
plt.show()