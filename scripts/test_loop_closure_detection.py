#######################
# Fix import issues   #
import sys            #
sys.path.append(".")  #
sys.path.append("..") #
#######################

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from threading import *

import src.dataloader as dataloader
import src.icp as icp
import src.loop_closure_detection as loop_closure_detection
import src.pose_graph as pose_graph
import src.pose_graph_optimization as pose_graph_optimization
import src.produce_occupancy_grid as produce_occupancy_grid
import src.utils as utils
import src.visualization as visualization

# This is the script that calls everything else. It will take in various command line arguments (such as filename, parameters), and runs SLAM

print("Loading the data...")
odometry, lidar_points, images = dataloader.parse_lcm_log("./data/EECS_6", load_images=True, image_stop=np.inf)
print("Done!")

dataloader.create_results_file_structure()

start = 11
dpi = 100
cell_width = 0.05
image_rate = 2

odometry = odometry[start:]
lidar_points = lidar_points[start:]
images = images[start:]

pg = pose_graph.PoseGraph(odometry)

print("Detecting loop closures")
loop_closure_detection.detect_images_direct_similarity(pg, lidar_points, images, image_rate=image_rate, min_dist_along_path=5, save_dists=True, save_matches=True, n_matches=20, image_err_thresh=2500)

fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=dpi)
visualization.draw_pose_graph(ax, pg)
visualization.draw_path(ax, odometry[:,:2])
# plt.savefig("results/init_pose_graph.png")
plt.show()

print("Optimizing pose graph...")
fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=dpi)
max_iters = 200
for iters in tqdm(range(max_iters)):
	pose_graph_optimization.pose_graph_optimization_step_sgd(pg)
	ax.cla()
	visualization.draw_pose_graph(ax, pg)
	visualization.draw_path(ax, pg.poses[:,:2])
	plt.draw()
	plt.pause(0.1)
	# plt.savefig("results/optim_fame%04d.png" % iters)
plt.close(fig)

print("Recorded %d poses. Creating occupancy grid..." % len(pg.poses))
og, (min_x, min_y) = produce_occupancy_grid.produce_occupancy_grid(pg.poses, lidar_points[:len(pg.poses)], cell_width, kHitOdds=20, kMissOdds=10)
print("Drawing occupancy grid...")
fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=dpi)
visualization.draw_occupancy_grid(ax, og, cell_size=cell_width, origin_location=np.array([min_x, min_y]))
# plt.savefig("results/final_map_og.png")
plt.show()

fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=dpi)
visualization.draw_point_map(ax, pg.poses, lidar_points[:len(pg.poses)])
# plt.savefig("results/final_map_points.png")
plt.show()