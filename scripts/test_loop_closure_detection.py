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

# This is the script that calls everything else. It will take in various command line arguments (such as filename, parameters), and runs SLAM

print("Loading the data...")
odometry, lidar_points, images = dataloader.parse_lcm_log("./data/EECS_3", load_images=True, image_stop=np.inf)
print("Done!")

start = 11
dpi = 100
cell_width = 0.05

odometry = odometry[start:]
lidar_points = lidar_points[start:]
images = images[start:]

pg = pose_graph.PoseGraph(odometry)

print("Detecting loop closures")
loop_closure_detection.detect_images_direct_similarity(pg, lidar_points, images, min_dist_along_path=5)

fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=dpi)
visualization.draw_pose_graph(ax, pg)
visualization.draw_path(ax, odometry[:,:2])
plt.show()

fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=dpi)
# visualization.draw_pose_graph(ax, pg)
# visualization.draw_path(ax, corrected_poses[:,:2])
# plt.draw()
# plt.pause(0.1)

print("Optimizing pose graph...")
iters = 0
max_iters = 100
while True:
	iters += 1
	pose_graph_optimization.pose_graph_optimization_step(pg, learning_rate=1/float(iters))
	ax.cla()
	visualization.draw_pose_graph(ax, pg)
	visualization.draw_path(ax, pg.poses[:,:2])
	# plt.draw()
	# plt.pause(0.1)
	plt.savefig("optim_fame%04d.png" % iters)
	print(iters)

	if iters >= max_iters:
		plt.close(fig)
		break

print("Recorded %d poses. Creating occupancy grid..." % len(pg.poses))
og, (min_x, min_y) = produce_occupancy_grid.produce_occupancy_grid(pg.poses, lidar_points[:len(pg.poses)], cell_width, kHitOdds=20, kMissOdds=10)
print("Drawing occupancy grid...")
fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=dpi)
visualization.draw_occupancy_grid(ax, og, cell_size=cell_width, origin_location=np.array([min_x, min_y]))
plt.savefig("final_map_og.png")
plt.show()

fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=dpi)
visualization.draw_point_map(ax, pg.poses, lidar_points[:len(pg.poses)])
plt.savefig("final_map_points.png")
plt.show()