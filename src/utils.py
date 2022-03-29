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