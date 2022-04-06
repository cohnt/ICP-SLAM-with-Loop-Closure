import numpy as np
import scipy.spatial

import src.icp as icp
import src.utils as utils

def detect_proximity(pose_graph, lidar_points, min_dist_along_path=15, max_dist=5, err_thresh=110):
	pairwise_dists = scipy.spatial.distance.cdist(pose_graph.poses[:,:2], pose_graph.poses[:,:2])
	dist_traveled = np.cumsum(np.diag(pairwise_dists, k=1))
	dist_traveled = np.append([0],dist_traveled)

	matches = []
	for i in range(len(pose_graph.poses)):
		start_idx = np.searchsorted(dist_traveled, dist_traveled[i]+min_dist_along_path, side="right")
		if start_idx >= len(pose_graph.poses):
			break
		closest = start_idx + np.argmin(pairwise_dists[i, start_idx:])
		if pairwise_dists[i, closest] <= max_dist:
			matches.append([i, closest])

	points_used = set()
	for i, j in matches:
		if (i not in points_used) and (j not in points_used):
			# estimated_tf = utils.pose_to_mat(pose_graph.poses[j] - pose_graph.poses[i])
			estimated_tf = np.eye(3)
			pc_prev = np.c_[lidar_points[i], np.ones(len(lidar_points[i]))]
			pc_current = np.c_[lidar_points[j], np.ones(len(lidar_points[j]))]
			tfs, error = icp.icp(pc_current, pc_prev, init_transform=estimated_tf, max_iters=100, epsilon=0.05)
			if error < err_thresh:
				print("%d %d %f" % (i, j, error))
				pose_graph.add_constraint(i, j, tfs[-1])
				points_used.add(i)
				points_used.add(j)