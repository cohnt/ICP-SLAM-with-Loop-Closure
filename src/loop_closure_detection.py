import numpy as np
import scipy.spatial
import cv2

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

def detect_images_direct_similarity(pose_graph, lidar_points, images, image_rate=1, min_dist_along_path=5, err_thresh=150, n_matches=10):
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
	keypoints = []
	descriptors = []
	# sift = cv2.SIFT_create()
	orb = cv2.ORB_create()
	for i in range(0, len(greys), image_rate):
		# kp, des = sift.detectAndCompute(greys[i], None)
		kp, des = orb.detectAndCompute(greys[i], None)
		keypoints.append(kp)
		descriptors.append(des)

	print("Matching keypoints")
	dist_mat = np.full((len(descriptors), len(descriptors)), np.inf)
	for i in range(0, len(descriptors)):
		for j in range(start_idx[i], len(descriptors)):
			bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
			matches = bf.match(descriptors[i], descriptors[j])
			matches = sorted(matches, key = lambda x:x.distance)

			if len(matches) < n_matches:
				continue

			dist_mat[i,j] = np.sum([match.distance for match in matches[:n_matches]])

	import matplotlib.pyplot as plt
	plt.imshow(dist_mat)
	plt.show()
	threshed = dist_mat < err_thresh
	plt.imshow(threshed)
	plt.show()

	good_matches = []
	for j in range(dist_mat.shape[1]):
		i = np.argmin(dist_mat[:,j])
		if dist_mat[i,j] < err_thresh:
			good_matches.append([i, j])
	print(good_matches)

	points_used = set()
	for i, j in good_matches:
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


	# import pdb
	# pdb.set_trace()



	# i = 0
	# j = 50

	# # bf = cv2.BFMatcher()
	# # matches = bf.knnMatch(des[i],des[j],k=1)

	# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	# matches = bf.match(descriptors[i], descriptors[j])
	# matches = sorted(matches, key = lambda x:x.distance)

	# # GOOD
	# # 7.0
	# # 9.0
	# # 11.0
	# # 11.0
	# # 11.0
	# # 11.0
	# # 12.0
	# # 12.0
	# # 12.0
	# # 12.0

	# # BAD
	# # 25.0
	# # 26.0
	# # 26.0
	# # 26.0
	# # 27.0
	# # 27.0
	# # 28.0
	# # 28.0
	# # 28.0
	# # 29.0



	# for m in matches[:10]:
	# 	print(m.distance)

	# img3 = cv2.drawMatches(greys[i],keypoints[i],greys[j],keypoints[j],matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

	# # for m in matches:
	# # 	m = m[0]
	# # 	print(m.distance)

	# # img3 = cv2.drawMatchesKnn(greys[i],keypoints[i],greys[j],keypoints[j],good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
	
	# import matplotlib.pyplot as plt
	# plt.imshow(img3)
	# plt.show()