import numpy as np
import src.utils as utils

def pose_graph_optimization_step(pose_graph, learning_rate=1, loop_closure_uncertainty=0.2):
	gamma = np.full(3, np.inf)
	N = pose_graph.graph.number_of_nodes()
	M = np.zeros((N, 3))

	# Approximate M
	for a, b, tf in pose_graph.graph.edges(data="object"):
		if np.abs(a - b) == 1:
			# The pose graph optimization does not consider successive points as constraints
			continue
		sigma = np.eye(3) * loop_closure_uncertainty
		R = construct_R(pose_graph, a)
		W = np.linalg.inv(R @ sigma @ R.T)
		for i in range(a+1, b+1):
			dW = np.diag(W)
			M[i] = M[i] + dW
			if np.dot(gamma, gamma) > np.dot(dW, dW):
				gamma = dW

	# Modified SGD
	for a, b, tf in pose_graph.graph.edges(data="object"):
		if np.abs(a - b) == 1:
			# The pose graph optimization does not consider successive points as constraints
			continue
		sigma = np.eye(3) * loop_closure_uncertainty
		R = construct_R(pose_graph, a)
		Pb_new = utils.pose_to_mat(pose_graph.poses[a]) @ tf
		r = utils.mat_to_pose(Pb_new) - pose_graph.poses[b]
		r[2] = r[2] % (2 * np.pi)
		d = 2 * np.linalg.inv(R.T @ sigma @ R) @ r.reshape(-1, 1)

		for j in range(3):
			alpha = 1 / gamma[j]
			alpha *= learning_rate
			total_weight = np.sum(1 / M[a+1:b+1,j])
			beta = (b - a) * d[j,0] * alpha
			if np.abs(beta) > np.abs(r[j]):
				beta = r[j]
			dpose = 0
			for i in range(a+1, N):
				if i <= b:
					dpose = dpose + (beta / M[i,j] / total_weight)
				pose_graph.poses[i,j] = pose_graph.poses[i,j] + dpose

def construct_R(pose_graph, idx):
	theta =pose_graph.poses[idx][2]
	c = np.cos(theta)
	s = np.sin(theta)
	R = np.array([
		[c, -s, 0],
		[s, c, 0],
		[0, 0, 1]
	])
	return R