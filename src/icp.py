import numpy as np


def get_closest_point(point, pc):
	distances = np.sum((pc - point) ** 2, axis=1)
	return np.argmin(distances)


def get_correspondences(pc1, pc2):
	correspondences = np.zeros(pc1.shape[0], dtype=int)
	for i in range(pc2.shape[0]):
		correspondences[i] = get_closest_point(pc1[i], pc2)

	return correspondences


def get_transform(pc1, pc2):
	pc1_avg = np.sum(pc1, axis=0) / pc1.shape[0]
	pc2_avg = np.sum(pc2, axis=0) / pc2.shape[0]

	X = pc1[:, 0:2] - pc1_avg[:2]
	Y = pc2[:, 0:2] - pc2_avg[:2]

	S = X.T @ Y
	U, sigma, V_t = np.linalg.svd(S, full_matrices=False)
	V = V_t.T

	mid_thingy = np.eye(2)
	mid_thingy[1, 1] = np.linalg.det(V @ U.T)
	R = V @ mid_thingy @ U.T
	t = pc2_avg[0:2].reshape((-1, 1)) - R @ pc1_avg[:2].reshape((-1, 1))

	transformed = np.eye(3)
	transformed[0:2, 0:2] = R
	transformed[0, 2] = t[0, 0]
	transformed[1, 2] = t[1, 0]

	return transformed


def get_error(pc1, pc2, correspondences):
	return np.sum((pc1 - pc2[correspondences]) ** 2)


def icp_iteration(pc1, pc2, previous_transform):
	pc1_transformed = np.dot(previous_transform, pc1.T).T
	correspondences = get_correspondences(pc1_transformed, pc2)
	trans_mat = get_transform(pc1_transformed, pc2[correspondences])
	trans_mat = trans_mat @ previous_transform
	error = get_error(pc1_transformed, pc2, correspondences)
	return trans_mat, correspondences, error


def icp(pc1, pc2, init_transform=np.eye(3), epsilon=0.01):
	# Runs ICP to estimate the transformation from pc1 to pc2. Optional parameter
	# init_transform is a 3x3 matrix in SE(2), that provides the initialization
	# for ICP. Returns a 3x3 matrix in SE(2) that estimates the final transformation
	# points in point clouds are expected to be homogeneous coordinates

	transforms = [init_transform]
	iteration = 0
	max_iters = 100

	while True:
		next_transform, correspondences, error = icp_iteration(pc1, pc2, transforms[-1])
		transforms.append(next_transform)
		print(error)
		if error < epsilon:
			return transforms
		if iteration > max_iters:
			return transforms
		iteration += 1

