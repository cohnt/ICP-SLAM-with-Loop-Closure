import numpy as np

def odom_change_to_mat(delta):
	# Given a change in position delta, as a 3-tuple (dx, dy, dtheta),
	# return a 3x3 transformation matrix in SE(2) that encodes the transforamtion
	dx, dy, dtheta = delta

	c = np.cos(dtheta)
	s = np.sin(dtheta)
	
	mat = np.eye(3)
	mat[0,0] = c
	mat[0,1] = -s
	mat[1,0] = s
	mat[1,1] = c
	mat[0,2] = dx
	mat[1,2] = dy

	return mat

def invert_affine(mat):
	# Given an nxn homogeneous matrix, return its inverse
	new_mat = np.eye(mat.shape[0])
	new_mat[:-1,:-1] = mat[:-1,:-1].T
	new_mat[:-1,-1] = new_mat[:-1,:-1] @ mat[:-1,-1]
	return new_mat

def pose_to_mat(pose):
	return np.array([
		[np.cos(pose[2]), -np.sin(pose[2]), pose[0]],
		[np.sin(pose[2]), np.cos(pose[2]), pose[1]],
		[0, 0, 1]
	])

def mat_to_pose(mat):
	return np.array([mat[0,2], mat[1,2], np.arctan2(mat[1,0], mat[0,0])])

def homogenize(vecs):
	# TODO
	pass

def unhomogenize(vecs):
	# TODO
	pass