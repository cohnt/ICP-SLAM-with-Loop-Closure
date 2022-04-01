import numpy as np

def get_closest_point(point, pc):
	distances = np.sum((pc - point) ** 2, axis=0)
	return np.argmin(distances)


def get_correspondences(pc1, pc2):
	correspondences = np.zeros(pc1.shape[0])
	for i in range(pc2.shape[0]):
		correspondences[i] = get_closest_point(pc1[i], pc2)

	return correspondences


def get_transform(pc1, pc2, correstpondences):
	return None


def get_error(pc1, pc2, correstpondences):
	return np.sum((pc1 - pc2[correstpondences]) ** 2)


def icp(pc1, pc2, init_transform=np.eye(3), epsilon=0.01):
	# Runs ICP to estimate the transformation from pc1 to pc2. Optional parameter
	# init_transform is a 3x3 matrix in SE(2), that provides the initialization
	# for ICP. Returns a 3x3 matrix in SE(2) that estimates the final transformation

	# Convert to Homogeneous coordinates
	pc1 = np.hstack((pc1, np.zeros(pc1.shape[0], 1)))
	pc2 = np.hstack((pc2, np.zeros(pc2.shape[0], 1)))

	transforms = init_transform
	iteration = 0

	while True:
		transformed = np.dot(transforms[-1], pc1.T).T
		correspondences = get_correspondences(transformed, pc2)
		transformation = get_transform(transformed, pc2, correspondences)
		transforms = np.append(transforms, transformation)
		if get_error(transformed, pc2, correspondences) < epsilon:
			return transforms
		iteration += 1

