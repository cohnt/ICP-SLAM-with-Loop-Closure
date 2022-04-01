import numpy as np

def get_closest_point(point, pc):
	distances = np.sum((pc - point) ** 2, axis=1)
	return np.argmin(distances)


def get_correspondences(pc1, pc2):
	correspondences = np.zeros(pc1.shape[0], dtype=int)
	for i in range(pc2.shape[0]):
		correspondences[i] = get_closest_point(pc1[i], pc2)

	return correspondences


def get_transform(pc1, pc2, correspondences):
	pc2 = pc2[correspondences]
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
	t = pc2_avg - R @ pc1_avg[:2].reshape(-1,1)

	transformed = np.eye(3)
	transformed[0:2, 0:2] = R
	transformed[2, 0] = t[0,0]
	transformed[2, 1] = t[1,0]

	return transformed


def get_error(pc1, pc2, correspondences):
	return np.sum((pc1 - pc2[correspondences]) ** 2)


def icp(pc1, pc2, init_transform=np.eye(3), epsilon=0.01):
	# Runs ICP to estimate the transformation from pc1 to pc2. Optional parameter
	# init_transform is a 3x3 matrix in SE(2), that provides the initialization
	# for ICP. Returns a 3x3 matrix in SE(2) that estimates the final transformation

	transforms = [init_transform]
	iteration = 0
	max_iters = 100

	while True:
		transformed = np.dot(transforms[-1], pc1.T).T
		correspondences = get_correspondences(transformed, pc2)
		transformation = get_transform(transformed, pc2, correspondences)
		transforms.append(transformation)
		print(get_error(transformed, pc2, correspondences))
		if get_error(transformed, pc2, correspondences) < epsilon:
			return transforms
		if iteration > max_iters:
			return transforms
		iteration += 1

