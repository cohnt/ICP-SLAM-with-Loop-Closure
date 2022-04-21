#######################
# Fix import issues   #
import os
import sys            #
import traceback      #
sys.path.append(".")  #
sys.path.append("..") #
#######################

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse

try:
	import src.dataloader as dataloader
	import src.icp as icp
	import src.loop_closure_detection as loop_closure_detection
	import src.pose_graph as pose_graph
	import src.pose_graph_optimization as pose_graph_optimization
	import src.produce_occupancy_grid as produce_occupancy_grid
	import src.utils as utils
	import src.visualization as visualization
except Exception as e:
	print("Failed to import source files. Make sure this script is run from the root of the repository, or from within the scripts folder!")
	traceback.print_exc()
	exit(1)

dataloader.create_results_file_structure()
with open("results/cmd.txt", "w") as cmd_file:
	cmd_file.write("Test run with the command:\n")
	cmd_file.write("python3 " + " ".join(sys.argv))

# Overall flow for the main function, when used to do everything in full
# 1) Load the dataset
# 2) Match poses with ICP, create pose graph
# 3) Find loop closures with image keypoints, and add to pose graph
# 4) Optimize pose graph

# Ways to change the flow
# -Use a prelearned pose graph as input to step 3
# -Use a prelearned pose graph as input to step 4
# -Skip ICP and just use odometry for step 2
# -Exit early at any stage

# Parameters that control program flow
# --start: "scan_matching", "loop_closure", "optimization"
#   Determines the first step undertaken by the program. If not scan_matching, a pose
#   graph must be loaded with the option --pose-graph
# --end: "scan_matching", "loop_closure", "optimization"
#   Determines the final step undertaken by the program
# --skip-icp
#   If present, the program doesn't run ICP, and simply matches scans with the raw odometry

# Other parameters
# --dataset [str]
#   The folder containing the data for this run
# --pose-graph [str]
#   The file containing the pose graph that will be used as input to step 3 or 4, depending
#   on the value for --start
# --n_jobs [int]
#   Number of processors used for multithreaded portions of the code.
#   Set to -1 to use all processors

parser = argparse.ArgumentParser()
parser.add_argument("dataset",
 help="Path to the dataset folder."
)
parser.add_argument("--program-start", default="scan_matching", choices=["scan_matching", "loop_closure", "optimization"],
 help="Control the starting stage of the program. Should be \"scan_matching\", \"loop_closure\",\
 or \"optimization\". If it's not \"scan_matching\", then an input pose graph must be specified using\
 the --pose-graph option."
)
parser.add_argument("--program-end", default="optimization", choices=["scan_matching", "loop_closure", "optimization"],
 help="Control the ending stage of the program. Should be \"scan_matching\", \"loop_closure\",\
 or \"optimization\", and must match or succeed the stage specified in --program_start."
)
parser.add_argument("--skip-icp", action="store_true",
 help="Skip matching scans with ICP, and just trust the given odometry. Only relevant if --program_start is\
 \"scan_matching\"."
)
parser.add_argument("--icp-max-iters", default=100, type=int,
 help="Maximum number of ICP iterations to run (default: 100)."
)
parser.add_argument("--icp-epsilon", default=0.05, type=float,
 help="ICP epsilon value. If the error goes below this value, the ICP stops (default: 0.05)."
)
parser.add_argument("--pose-graph",
 help="Filename of the pose graph to use at the starting stage. Only relevant if --program_start is\
 \"loop_closure\" or \"optimization\"."
)
parser.add_argument("--n-jobs", default=-1, type=int,
 help="The number of processors to use for multithreaded protions of the code. Set to -1 to use all\
 processors."
)
parser.add_argument("--dataset-start", default=0, type=int,
 help="The first frame to be used in the SLAM system from the given dataset. (All preceeding frames\
 are discarded."
)
parser.add_argument("--dataset-end", type=int,
 help="The last frame to be used in the SLAM system from the given dataset. (All succeeding frames\
 are discarded."
)
parser.add_argument("--figure-dpi", default=100, type=int,
 help="The dpi of the saved matplotlib figures (default: 100). See also: --figure-width and --figure-height.\
 The default values for these parameters produces 1920x1080 figures."
)
parser.add_argument("--figure-width", default=19.2, type=float,
 help="Width of the saved matplotlib figures, in inches (default: 19.2). See also: --figure-dpi and\
 --figure-height. The default values for these parameters produces 1920x1080 figures."
)
parser.add_argument("--figure-height", default=10.8, type=float,
 help="Height of the saved matplotlib figures, in inches (default: 10.8). See also: --figure-dpi and\
 --figure-width. The default values for these parameters produces 1920x1080 figures."
)
parser.add_argument("--image-downsample", default=1, type=int,
 help="The proporiton of the images to use for loop closure identification (default: 1). 1 will take every\
 image, 2 will take every other image, 3 will take every third image, et cetera."
)
parser.add_argument("--image-match-error", default=2500, type=float,
 help="The error threshold for two images to be considered a match (default: 2500)."
)
parser.add_argument("--loop-closure-icp-error", default=30, type=float,
 help="The ICP error threshold to include a pair of matched images in the pose graph (default: 30)."
)
parser.add_argument("--keypoint-n-matches", default=20, type=int,
 help="The number of keypoints to match across images when detecting loop closures (default: 20)."
)
parser.add_argument("--cell-width", default=0.1, type=float,
 help="The width of the grid cells in the produced occupancy grid maps, in meters (default: 0.1)."
)
parser.add_argument("--hit-odds", default=5, type=int,
 help="The hit odds used by the occupancy grid mapper (default: 5). Should be at least twice as large\
 as --miss-odds."
)
parser.add_argument("--miss-odds", default=2, type=int,
 help="The miss odds used by the occupancy grid mapper (default 2). Should be at most half as large as\
 --miss-odds."
)
parser.add_argument("--produce-odometry-map", action="store_true",
 help="Generate and save maps from the initial odometry estimates. Only really useful as a baseline."
)
parser.add_argument("--skip-occupancy-grid", action="store_true",
 help="Skip producing the occupancy grid maps."
)
parser.add_argument("--save-icp-images", action="store_true",
 help="Save images of each ICP iteration. This is very slow for large maps."
)
parser.add_argument("--image-pointcloud-downsample", default=10, type=int,
 help="Downsampling applied to point clouds before saving images (default: 10). 1 will take all points,\
 2 will take, every other point, 3 will take every third point, et cetera."
)
parser.add_argument("--min-dist-along-path", default=5, type=int,
 help="The minimum traveled distance to add a loop closure between two poses (default: 5)."
)
parser.add_argument("--no-save-matches", action="store_true",
 help="Don't store images of each loop closure match identified. (Saves the images by default.)"
)
parser.add_argument("--no-save-dist-mat", action="store_true",
 help="Don't store a plot of the loop closure image distance matrix. (Saves the plot by default.)"
)
parser.add_argument("--save-map-files", action="store_true",
 help="Expore occupancy grid maps as .map files, for use with the robot."
)
parser.add_argument("--optimization-max-iters", default=50, type=int,
 help="Maximum number of iterations for the pose graph optimization (default: 50)."
)
parser.add_argument("--occupancy-grid-mle", action="store_true",
 help="Take the MLE of each grid square in the occupancy grid (either obstacle or not). Grid squares\
 with no information (i.e. the probability is precisely 0.5) will remain unchanged."
)
parser.add_argument("--manual-loop-closures",
 help="File containing manual loop closure annotations. File is just a text file where each row contains two\
 numbers corresponding to the images that a loop closure constraint should be added to."
)
parser.add_argument("--icp-recompute", action="store_true",
 help="Re-estimate the robot orientations after pose graph optimization using ICP."
)

args = parser.parse_args()

program_start = args.program_start
program_end = args.program_end
program_use_icp = not bool(args.skip_icp)

make_odometry_maps = bool(args.produce_odometry_map)

dataset_fname = args.dataset
dataset_start = args.dataset_start
dataset_end = args.dataset_end if args.dataset_end else np.inf
dpi = args.figure_dpi
figure_width = args.figure_width
figure_height = args.figure_height
figsize = (figure_width, figure_height)
cell_width = args.cell_width
image_rate = args.image_downsample
n_jobs = args.n_jobs
kHitOdds = args.hit_odds
kMissOdds = args.miss_odds
n_matches = args.keypoint_n_matches
image_err_thresh = args.image_match_error
icp_max_iters = args.icp_max_iters
icp_epsilon = args.icp_epsilon
icp_err_thresh = args.loop_closure_icp_error
save_icp_images = bool(args.save_icp_images)
image_pointcloud_downsample = args.image_pointcloud_downsample
pose_graph_fname = args.pose_graph
min_dist_along_path = args.min_dist_along_path
loop_closure_icp_error = args.loop_closure_icp_error
save_matches = not bool(args.no_save_matches)
save_dist_mat = not bool(args.no_save_dist_mat)
save_map_files = bool(args.save_map_files)
optimization_max_iters = args.optimization_max_iters
skip_occupancy_grid = args.skip_occupancy_grid
occupancy_grid_mle = bool(args.occupancy_grid_mle)
manual_annotation_file = args.manual_loop_closures if args.manual_loop_closures else None
icp_recompute = bool(args.icp_recompute)

if program_start != "scan_matching":
	if pose_graph_fname is None:
		print("Error: If starting after scan matching, a pose graph must be passed in using the command line option --pose-graph.")
		exit(1)

print("Loading the data")
odometry, lidar_points, images = dataloader.parse_lcm_log(dataset_fname, load_images=True, image_stop=dataset_end, n_jobs=n_jobs)

odometry = odometry[dataset_start:]
lidar_points = lidar_points[dataset_start:]
images = images[dataset_start:]

if make_odometry_maps:
	visualization.gen_and_save_map(odometry, lidar_points, "odometry", cell_width, kHitOdds, kMissOdds, dpi, figsize=figsize,
		save_map_files=save_map_files, skip_occupancy_grid=skip_occupancy_grid, mle=occupancy_grid_mle)

if program_start == "scan_matching":
	if program_use_icp:
		print("Aligning poses with ICP")
		raw_tfs = odometry[1:] - odometry[:-1]
		parallel = Parallel(n_jobs=n_jobs, verbose=0, backend="loky")
		tfs, errs = zip(*parallel(delayed(icp.icp)(
			np.c_[lidar_points[i],np.ones(len(lidar_points[i]))],
			np.c_[lidar_points[i-1],np.ones(len(lidar_points[i-1]))],
			init_transform=utils.pose_to_mat(odometry[i] - odometry[i-1]),
			max_iters=icp_max_iters,
			epsilon=icp_epsilon
		) for i in tqdm(range(1, len(odometry)))))

		corrected_poses = np.zeros((len(odometry), 3))
		corrected_poses[0] = odometry[0]
		for i in range(1, len(odometry)):
			real_tf = tfs[i-1][-1]
			real_prev_pose = utils.pose_to_mat(corrected_poses[i-1])
			real_pose = real_prev_pose @ real_tf
			real_odom = utils.mat_to_pose(real_pose)
			corrected_poses[i] = real_odom

		if save_icp_images:
			print("Saving ICP images")
			fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
			for i in tqdm(range(len(odometry))):
				pc_raw = np.c_[lidar_points[i],np.ones(len(lidar_points[i]))]
				pc = (utils.pose_to_mat(corrected_poses[i]) @ pc_raw.T).T
				pc = pc[::image_pointcloud_downsample]
				ax.scatter(pc[:,0], pc[:,1], color="red", s=0.1)

				visualization.draw_path(ax, corrected_poses[:i])
				ax.set_aspect("equal")
				plt.savefig("results/icp_frame%04d.png" % i)
			plt.close(fig)

		visualization.gen_and_save_map(corrected_poses, lidar_points, "icp", cell_width, kHitOdds, kMissOdds, dpi,
			figsize=figsize, save_map_files=save_map_files, skip_occupancy_grid=skip_occupancy_grid, mle=occupancy_grid_mle)
		pg = pose_graph.PoseGraph(corrected_poses)
		pg.save("results/icp_pose_graph.pickle")
		pg.export_g2o("results/icp_pose_graph.g2o")
	else:
		# (Not using ICP)
		corrected_poses = odometry.copy()
		pg = pose_graph.PoseGraph(corrected_poses)
		pg.save("results/odometry_pose_graph.pickle")
		pg.export_g2o("results/odometry_pose_graph.g2o")

if program_end == "scan_matching":
	exit(0)

if program_start != "scan_matching":
	pg = pose_graph.PoseGraph(None)
	pg.load(pose_graph_fname)

if program_start == "scan_matching" or program_start == "loop_closure":
	if manual_annotation_file is None:
		print("Detecting loop closures")
		loop_closure_detection.detect_images_direct_similarity(pg, lidar_points, images, min_dist_along_path=min_dist_along_path,
			save_dists=save_dist_mat, save_matches=save_matches, image_rate=image_rate, n_matches=n_matches, image_err_thresh=image_err_thresh,
			icp_err_thresh=loop_closure_icp_error)
	else:
		print("Adding manual loop closure constraints...")
		matches = np.loadtxt(manual_annotation_file, dtype=int)
		for match in matches:
			i, j = match
			estimated_tf = np.eye(3)
			pc_i = np.c_[lidar_points[i], np.ones(len(lidar_points[i]))]
			pc_j = np.c_[lidar_points[j], np.ones(len(lidar_points[j]))]
			tfs, error = icp.icp(pc_i, pc_j, init_transform=estimated_tf, max_iters=100, epsilon=0.05)
			if error < 30:
				pg.add_constraint(i, j, tfs[-1])

	pg.save("results/loop_closure_pose_graph.pickle")
	pg.export_g2o("results/loop_closure_pose_graph.g2o")

	fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
	visualization.draw_pose_graph(ax, pg)
	visualization.draw_path(ax, pg.poses[:, :2])
	ax.set_aspect("equal")
	plt.savefig("results/init_pose_graph.png")


if program_end == "loop_closure":
	exit(0)

if program_start == "scan_matching" or program_start == "loop_closure" or program_start == "optimization":
	print("Optimizing pose graph")
	fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
	for iters in tqdm(range(optimization_max_iters)):
		pose_graph_optimization.pose_graph_optimization_step_sgd(pg, learning_rate=1/float(iters+1))
		ax.cla()
		visualization.draw_pose_graph(ax, pg, draw_orientation=False)
		visualization.draw_path(ax, pg.poses[:,:2])
		ax.set_aspect("equal")
		plt.savefig("results/optim_fame%04d.png" % iters)
	plt.close(fig)
	print("Recomputing pose orientations")
	pose_graph_optimization.recompute_pose_graph_orientation(pg, lidar_points, icp_max_iters, icp_epsilon, n_jobs, icp_recompute=icp_recompute)

	visualization.gen_and_save_map(pg.poses, lidar_points, "final", cell_width, kHitOdds, kMissOdds, dpi, figsize=figsize,
		save_map_files=save_map_files, skip_occupancy_grid=skip_occupancy_grid, mle=occupancy_grid_mle)
	pg.save("results/optim.pickle")
	pg.export_g2o("results/optim.g2o")

if program_end == "optimization":
	exit(0)
