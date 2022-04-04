import numpy as np


def get_closest_point(point, pc):
	# Returns the closest point in a point cloud to a point
	distances = np.sum((pc - point) ** 2, axis=1)
	return np.argmin(distances)


def get_correspondences(pc1, pc2):
	# Gets correspondences between two point clouds
	# Returns n length numpy array where n is the number of points in pc1
	# correspondences[i] is the point in pc2 that corresponds to point i in pc1

	correspondences = np.zeros(pc1.shape[0], dtype=int)
	for i in range(pc2.shape[0]):
		correspondences[i] = get_closest_point(pc1[i], pc2)

	return correspondences


def get_transform(pc1, pc2):
	# Calculates the transformation matrix between two point clouds
	# Assumes that pc1[i] corresponds to pc2[i]

	pc1_avg = np.sum(pc1[:, 0:2], axis=0) / pc1.shape[0]
	pc2_avg = np.sum(pc2[:, 0:2], axis=0) / pc2.shape[0]

	X = (pc1[:, 0:2] - pc1_avg).T
	Y = (pc2[:, 0:2] - pc2_avg).T

	S = X @ Y.T
	U, sigma, V_t = np.linalg.svd(S)
	V = V_t.T

	mid_thingy = np.eye(2)
	mid_thingy[1, 1] = np.linalg.det(V @ U.T)
	R = V @ mid_thingy @ U.T
	t = pc2_avg.reshape((-1, 1)) - R @ pc1_avg.reshape((-1, 1))

	transformation_matrix = np.eye(3)
	transformation_matrix[0:2, 0:2] = R
	transformation_matrix[0, 2] = t[0, 0]
	transformation_matrix[1, 2] = t[1, 0]

	return transformation_matrix


def get_error(pc1, pc2):
	# Calculates the error between two point clouds
	# Assumes that pc1[i] corresponds to pc2[i]
	return np.sum((pc1 - pc2) ** 2)


def icp_iteration(pc1, pc2, previous_transform):
	# Performs a single iteration of icp
	# Assumes pc1 and pc2 are the original point clouds
	# previous_transform is the transformation matrix generated by the previous round

	pc1_transformed = np.dot(previous_transform, pc1.T).T
	correspondences = get_correspondences(pc1_transformed, pc2)
	trans_mat = get_transform(pc1_transformed, pc2[correspondences])
	error = get_error(pc1_transformed, pc2[correspondences])
	return trans_mat, correspondences, error


def icp(pc1, pc2, init_transform=np.eye(3), epsilon=0.01):
	# Runs ICP to estimate the transformation from pc1 to pc2. Optional parameter
	# init_transform is a 3x3 matrix in SE(2), that provides the initialization
	# for ICP. Returns a 3x3 matrix in SE(2) that estimates the final transformation
	# pc1 and pc2 are (n, 3) numpy arrays of homogenous coordinates

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

