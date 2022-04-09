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

def serialize_matches(matches):
	return [(match.queryIdx, match.trainIdx, match.distance) for match in matches]

def deserialize_matches(serialized_matches):
	return [cv2.DMatch(serialized_match[0], serialized_match[1], serialized_match[2]) for serialized_match in serialized_matches]

def find_keypoints(img):
	orb = cv2.ORB_create()
	kp, des = orb.detectAndCompute(img, None)
	return serialize_keypoints(kp, des)

def matchify(desc1, desc2, i, j, n_matches, approximate_match):
	if approximate_match:
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks=50)   # or pass empty dictionary
		flann = cv2.FlannBasedMatcher(index_params,search_params)
		desc1 = np.asarray(desc1, np.float32)
		desc2 = np.asarray(desc2, np.float32)
		matches = flann.match(desc1, desc2)
	else:
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		matches = bf.match(desc1, desc2)

	matches = sorted(matches, key = lambda x:x.distance)

	if len(matches) < n_matches:
		return np.inf

	return np.sum([match.distance for match in matches[:n_matches]]), serialize_matches(matches[:n_matches]), (i, j)

def detect_images_direct_similarity(pose_graph, lidar_points, images, image_rate=1, min_dist_along_path=5,
	                                image_err_thresh=125, n_matches=10, icp_err_thresh=30, save_dists=False,
	                                save_matches=False, n_jobs=-1, approximate_match=True):
	pairwise_dists = scipy.spatial.distance.cdist(pose_graph.poses[:,:2], pose_graph.poses[:,:2])
	dist_traveled = np.cumsum(np.diag(pairwise_dists, k=1))
	dist_traveled = np.append([0],dist_traveled)
	start_idx = np.array([
		np.searchsorted(dist_traveled, dist_traveled[i]+min_dist_along_path, side="right") for i in range(len(dist_traveled))
	])

	print("Converting to grayscale...")
	greys = [cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2GRAY) for image in images]

	print("Finding keypoints")
	parallel = Parallel(n_jobs=n_jobs, verbose=0, backend="loky")
	serialized_keypoints_list = parallel(delayed(find_keypoints)(greys[i]) for i in tqdm(range(0, len(greys), image_rate)))
	keypoints, descriptors = zip(*[deserialize_keypoints(serialized_keypoints) for serialized_keypoints in serialized_keypoints_list])

	print("Matching keypoints")
	parallel = Parallel(n_jobs=n_jobs, verbose=0, backend="loky")
	dist_mat_s, matched_keypoints_s, idx_s = zip(*parallel(delayed(matchify)(descriptors[i], descriptors[j], i, j, n_matches, approximate_match) for i in tqdm(range(0, len(descriptors))) for j in range(start_idx[i], len(descriptors))))

	matched_keypoints = [[None for _ in range(len(greys))] for _ in range(len(greys))]
	dist_mat = np.full((len(descriptors), len(descriptors)), np.inf)
	for idx in range(len(dist_mat_s)):
		i, j = idx_s[idx]
		dist_mat[i,j] = dist_mat_s[idx]
		matched_keypoints[i][j] = deserialize_matches(matched_keypoints_s[idx])

	print("Closest images keypoint match error %f" % np.min(dist_mat))
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
			good_matches.append([i, j])
			good_matches_keypoints.append(matched_keypoints[i][j])

	print("Aligning matched point clouds with ICP")
	parallel = Parallel(n_jobs=n_jobs, verbose=0, backend="loky")

	tfs, errs = zip(*parallel(delayed(icp.icp)(
		np.c_[lidar_points[i*image_rate],np.ones(len(lidar_points[i*image_rate]))],
		np.c_[lidar_points[j*image_rate],np.ones(len(lidar_points[j*image_rate]))],
		init_transform=np.eye(3),
		max_iters=100,
		epsilon=0.05
	) for i, j in tqdm(good_matches)))

	if save_matches:
		print("Saving matched images")
		iterable = tqdm(range(len(good_matches)))
	else:
		iterable = range(len(good_matches))
	for idx in iterable:
		i, j = good_matches[idx]
		old_i, old_j = i, j
		i *= image_rate
		j *= image_rate

		tf, error= tfs[idx][-1], errs[idx]
		# print(error)
		if error < icp_err_thresh:
			# print("%d %d %f" % (i, j, error))
			pose_graph.add_constraint(i, j, tf)
			if save_matches:
				match_img = cv2.drawMatches(greys[old_i],keypoints[old_i],greys[old_j],keypoints[old_j],good_matches_keypoints[idx],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
				fig, ax = plt.subplots()
				ax.imshow(match_img)
				plt.savefig("match_%d_%d_%f.png" % (i, j, dist_mat[old_i,old_j]))
				plt.close(fig)