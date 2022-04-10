#######################
# Fix import issues   #
import sys            #
sys.path.append(".")  #
sys.path.append("..") #
#######################

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

import src.dataloader as dataloader
import src.icp as icp
import src.loop_closure_detection as loop_closure_detection
import src.pose_graph as pose_graph
import src.pose_graph_optimization as pose_graph_optimization
import src.produce_occupancy_grid as produce_occupancy_grid
import src.utils as utils
import src.visualization as visualization

# This is the script that calls everything else. It will take in various command line arguments (such as filename, parameters), and runs SLAM

start = 11
end = np.inf
dpi = 100
cell_width = 0.25
image_rate = 2
n_jobs = -1 # Set the number of threads to use, or -1 to use all
kHitOdds = 30
kMissOdds = 10
n_matches = 20
image_err_thresh = 2500

print("Loading the data...")
odometry, lidar_points, images = dataloader.parse_lcm_log("./data/EECS_6", load_images=True, image_stop=end, n_jobs=n_jobs)
print("Done!")

odometry = odometry[start:]
lidar_points = lidar_points[start:]
images = images[start:]

visualization.gen_and_save_map(odometry, lidar_points, "odometry", cell_width, kHitOdds, kMissOdds, dpi)

print("Aligning poses with ICP...")
raw_tfs = odometry[1:] - odometry[:-1]
parallel = Parallel(n_jobs=n_jobs, verbose=0, backend="loky")
tfs, errs = zip(*parallel(delayed(icp.icp)(
	np.c_[lidar_points[i],np.ones(len(lidar_points[i]))],
	np.c_[lidar_points[i-1],np.ones(len(lidar_points[i-1]))],
	init_transform=utils.pose_to_mat(odometry[i] - odometry[i-1]),
	max_iters=100,
	epsilon=0.05
) for i in tqdm(range(1, len(odometry)))))

corrected_poses = np.array([odometry[0]])

print("Saving ICP images...")
# fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=dpi)
for i in tqdm(range(len(corrected_poses), len(odometry))):
	real_tf = tfs[i-1][-1]
	real_prev_pose = utils.pose_to_mat(corrected_poses[-1])
	real_pose = real_prev_pose @ real_tf
	real_odom = utils.mat_to_pose(real_pose)
	corrected_poses = np.vstack((corrected_poses, real_odom))

	pc_prev = np.c_[lidar_points[i],np.ones(len(lidar_points[i]))]
	pc_current = np.c_[lidar_points[i-1],np.ones(len(lidar_points[i-1]))]
	pc1 = (utils.pose_to_mat(corrected_poses[-2]) @ pc_prev.T).T
	pc2 = (utils.pose_to_mat(corrected_poses[-1]) @ pc_current.T).T
	ax.scatter(pc1[::10,0], pc1[::10,1], color="red", s=0.1)
	ax.scatter(pc2[::10,0], pc2[::10,1], color="blue", s=0.1)
	
	visualization.draw_path(ax, corrected_poses)
	ax.set_aspect("equal")
	
	# plt.draw()
	# plt.pause(0.1)
	plt.savefig("icp_frame%04d.png" % i)

# plt.close(fig)

visualization.gen_and_save_map(corrected_poses, lidar_points, "icp", cell_width, kHitOdds, kMissOdds, dpi)

pg = pose_graph.PoseGraph(corrected_poses)

print("Detecting loop closures")
loop_closure_detection.detect_images_direct_similarity(pg, lidar_points, images, min_dist_along_path=5, save_dists=True, save_matches=True, image_rate=image_rate, n_matches=n_matches, image_err_thresh=image_err_thresh)

# fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=dpi)
# visualization.draw_pose_graph(ax, pg)
# visualization.draw_path(ax, corrected_poses[:,:2])
# ax.set_aspect("equal")
# plt.show()

print("Optimizing pose graph...")
fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=dpi)
max_iters = 100
for iters in tqdm(range(max_iters)):
	pose_graph_optimization.pose_graph_optimization_step_sgd(pg, learning_rate=2/float(iters+1))
	ax.cla()
	visualization.draw_pose_graph(ax, pg)
	visualization.draw_path(ax, pg.poses[:,:2])
	# plt.draw()
	# plt.pause(0.1)
	plt.savefig("optim_fame%04d.png" % iters)
plt.close(fig)

visualization.gen_and_save_map(pg.poses, lidar_points, "final", cell_width, kHitOdds, kMissOdds, dpi)