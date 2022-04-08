import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
import cv2
from tqdm import tqdm
from joblib import Parallel, delayed

import src.icp as icp
import src.utils as utils

def detect_proximity(pose_graph, lidar_points, min_dist_along_path=2, max_dist=1, err_thresh=110):
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

	matches.reverse()
	points_used = set()
	for i, j in matches:
		if (i not in points_used) and (j not in points_used):
		# if True:
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

def serialize_keypoints(kp, des):
	return [(point.pt, point.size, point.angle, point.response, point.octave, point.class_id, desc) for point, desc in zip(kp, des)]

def deserialize_keypoints(serialized_keypoints):
	kp = [cv2.KeyPoint(x=point[0][0], y=point[0][1], size=point[1], angle=point[2],
	      response=point[3], octave=point[4], class_id=point[5]) for point in serialized_keypoints]
	des = np.array([point[6] for point in serialized_keypoints])
	return kp, des

def find_keypoints(img):
	orb = cv2.ORB_create()
	kp, des = orb.detectAndCompute(img, None)
	return serialize_keypoints(kp, des)

def detect_images_direct_similarity(pose_graph, lidar_points, images, image_rate=1, min_dist_along_path=5, image_err_thresh=125, n_matches=10, icp_err_thresh=30, save_dists=False, save_matches=False, n_jobs=-1):
	pairwise_dists = scipy.spatial.distance.cdist(pose_graph.poses[:,:2], pose_graph.poses[:,:2])
	dist_traveled = np.cumsum(np.diag(pairwise_dists, k=1))
	dist_traveled = np.append([0],dist_traveled)
	start_idx = np.array([
		np.searchsorted(dist_traveled, dist_traveled[i]+min_dist_along_path, side="right") for i in range(len(dist_traveled))
	])
	print(start_idx)

	print("Converting to grayscale...")
	greys = [cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2GRAY) for image in images]

	print("Finding keypoints")
	parallel = Parallel(n_jobs=n_jobs, verbose=0, backend="loky")
	serialized_keypoints_list = parallel(delayed(find_keypoints)(greys[i]) for i in tqdm(range(0, len(greys), image_rate)))
	keypoints, descriptors = zip(*[deserialize_keypoints(serialized_keypoints) for serialized_keypoints in serialized_keypoints_list])

	print("Matching keypoints")
	matched_keypoints = [[None for _ in range(len(greys))] for _ in range(len(greys))]
	dist_mat = np.full((len(descriptors), len(descriptors)), np.inf)
	for i in tqdm(range(0, len(descriptors))):
		for j in range(start_idx[i], len(descriptors)):
			bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
			matches = bf.match(descriptors[i], descriptors[j])
			matches = sorted(matches, key = lambda x:x.distance)

			if len(matches) < n_matches:
				continue

			dist_mat[i,j] = np.sum([match.distance for match in matches[:n_matches]])
			matched_keypoints[i][j] = matches[:n_matches]
	threshed = dist_mat < image_err_thresh

	if save_dists:
		fig, ax = plt.subplots()
		ax.imshow(dist_mat)
		plt.savefig("dist_mat.png")
		plt.close(fig)
		fig, ax = plt.subplots()
		ax.imshow(threshed)
		plt.savefig("dist_mat_threshed.png")
		plt.close(fig)

	good_matches = []
	good_matches_keypoints = []
	for j in range(dist_mat.shape[1]):
		i = np.argmin(dist_mat[:,j])
		if dist_mat[i,j] < image_err_thresh:
			print(dist_mat[i,j])
			good_matches.append([i, j])
			good_matches_keypoints.append(matched_keypoints[i][j])

	points_used = set()
	for idx in range(len(good_matches)):
		i, j = good_matches[idx]
		if (i not in points_used) or (j not in points_used):
			old_i, old_j = i, j
			i *= image_rate
			j *= image_rate
		# if True:
			# estimated_tf = utils.pose_to_mat(pose_graph.poses[j] - pose_graph.poses[i])
			estimated_tf = np.eye(3)
			pc_prev = np.c_[lidar_points[i], np.ones(len(lidar_points[i]))]
			pc_current = np.c_[lidar_points[j], np.ones(len(lidar_points[j]))]
			tfs, error = icp.icp(pc_current, pc_prev, init_transform=estimated_tf, max_iters=100, epsilon=0.05)
			if error < icp_err_thresh:
				print("%d %d %f" % (i, j, error))
				pose_graph.add_constraint(i, j, tfs[-1])
				points_used.add(i)
				points_used.add(j)
				if save_matches:
					match_img = cv2.drawMatches(greys[old_i],keypoints[old_i],greys[old_j],keypoints[old_j],good_matches_keypoints[idx],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
					fig, ax = plt.subplots()
					ax.imshow(match_img)
					plt.savefig("match_%d_%d_%f.png" % (i, j, dist_mat[old_i,old_j]))
					plt.close(fig)