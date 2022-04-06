import numpy as np
import scipy.spatial

import src.icp as icp
import src.utils as utils

def detect_proximity(pose_graph, lidar_points, min_dist_along_path=5, max_dist=2):
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

	for i, j in matches:
		pose_graph.add_constraint(i, j, np.eye(3))