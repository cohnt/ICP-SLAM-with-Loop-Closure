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
import src.manual_loop_closure as manual_loop_closure
import src.pose_graph as pose_graph
import src.pose_graph_optimization as pose_graph_optimization
import src.produce_occupancy_grid as produce_occupancy_grid
import src.store_pose_graph as store_pose_graph
import src.utils as utils
import src.visualization as visualization

# This is the script that calls everything else. It will take in various command line arguments (such as filename, parameters), and runs SLAM

start = 11
end = 200
dpi = 100
cell_width = 0.1
image_rate = 3

print("Loading the data...")
odometry, lidar_points, images = dataloader.parse_lcm_log("./data/EECS_3", load_images=True, image_stop=end)
print("Done!")

odometry = odometry[start:]
lidar_points = lidar_points[start:]
images = images[start:]

print("Producing raw odometry map...")
og, (min_x, min_y) = produce_occupancy_grid.produce_occupancy_grid(odometry, lidar_points, cell_width, kHitOdds=10, kMissOdds=10)
print("Drawing occupancy grid...")
fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=dpi)
visualization.draw_occupancy_grid(ax, og, cell_size=cell_width, origin_location=np.array([min_x, min_y]))
plt.savefig("odometry_map_og.png")
visualization.draw_path(ax, odometry)
plt.savefig("odometry_map_og_path.png")
plt.close(fig)

produce_occupancy_grid.save_image(og, "odometry_og.png")

fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=dpi)
visualization.draw_point_map(ax, odometry, lidar_points)
ax.set_aspect("equal")
plt.savefig("odometry_map_points.png")
visualization.draw_path(ax, odometry)
plt.savefig("odometry_map_points_path.png")
plt.close(fig)

print("Aligning poses with ICP...")
raw_tfs = odometry[1:] - odometry[:-1]
parallel = Parallel(n_jobs=-1, verbose=0, backend="loky")
tfs, errs = zip(*parallel(delayed(icp.icp)(
	np.c_[lidar_points[i],np.ones(len(lidar_points[i]))],
	np.c_[lidar_points[i-1],np.ones(len(lidar_points[i-1]))],
	init_transform=utils.pose_to_mat(odometry[i] - odometry[i-1]),
	max_iters=100,
	epsilon=0.05
) for i in tqdm(range(1, len(odometry)))))

corrected_poses = np.array([odometry[0]])

print("Saving ICP images...")
fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=dpi)
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
	# print("Frame %d" % i)

plt.close(fig)

print("Producing ICP-aligned map...")
og, (min_x, min_y) = produce_occupancy_grid.produce_occupancy_grid(corrected_poses, lidar_points, cell_width, kHitOdds=10, kMissOdds=10)
print("Drawing occupancy grid...")
fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=dpi)
visualization.draw_occupancy_grid(ax, og, cell_size=cell_width, origin_location=np.array([min_x, min_y]))
plt.savefig("icp_map_og.png")
visualization.draw_path(ax, corrected_poses)
plt.savefig("icp_map_og_path.png")
plt.close(fig)

produce_occupancy_grid.save_image(og, "icp_og.png")

fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=dpi)
visualization.draw_point_map(ax, corrected_poses, lidar_points)
ax.set_aspect("equal")
plt.savefig("icp_map_points.png")
visualization.draw_path(ax, corrected_poses)
plt.savefig("icp_map_points_path.png")
plt.close(fig)

pg = pose_graph.PoseGraph(corrected_poses)

print("Detecting loop closures")
loop_closure_detection.detect_images_direct_similarity(pg, lidar_points, images, min_dist_along_path=5, save_dists=True, save_matches=True, image_rate=image_rate)

# fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=dpi)
# visualization.draw_pose_graph(ax, pg)
# visualization.draw_path(ax, corrected_poses[:,:2])
# ax.set_aspect("equal")
# plt.show()

print("Optimizing pose graph...")
fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=dpi)
max_iters = 100
for iters in tqdm(range(max_iters)):
	pose_graph_optimization.pose_graph_optimization_step(pg, learning_rate=2/float(iters+1))
	ax.cla()
	visualization.draw_pose_graph(ax, pg)
	visualization.draw_path(ax, pg.poses[:,:2])
	# plt.draw()
	# plt.pause(0.1)
	plt.savefig("optim_fame%04d.png" % iters)
plt.close(fig)

print("Recorded %d poses. Creating occupancy grid..." % len(pg.poses))
og, (min_x, min_y) = produce_occupancy_grid.produce_occupancy_grid(pg.poses, lidar_points, cell_width, kHitOdds=10, kMissOdds=10)
print("Drawing occupancy grid...")
fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=dpi)
visualization.draw_occupancy_grid(ax, og, cell_size=cell_width, origin_location=np.array([min_x, min_y]))
plt.savefig("final_map_og.png")
visualization.draw_path(ax, pg.poses)
plt.savefig("final_map_og_path.png")
plt.show()

produce_occupancy_grid.save_image(og, "final_og.png")
produce_occupancy_grid.save_grid(og, "final.map", cell_width)

fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=dpi)
visualization.draw_point_map(ax, pg.poses, lidar_points)
ax.set_aspect("equal")
plt.savefig("final_map_points.png")
visualization.draw_path(ax, pg.poses)
plt.savefig("final_map_points_path.png")
plt.show()