import numpy as np
import src.utils as utils
from tqdm import tqdm
from joblib import Parallel, delayed
import src.icp as icp

def pose_graph_optimization_step_sgd(pose_graph, learning_rate=1, loop_closure_uncertainty=0.2):
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

def recompute_pose_graph_orientation(pose_graph, lidar_points, icp_max_iters, icp_epsilon, n_jobs):
	N = len(pose_graph.poses)
	for i in range(1, N-1):
		vec = pose_graph.poses[i+1][0:2] - pose_graph.poses[i][0:2]
		if np.linalg.norm(vec) > 0:
			vec = vec / np.linalg.norm(vec)
			pose_graph.poses[i][2] = np.arctan2(vec[1], vec[0])

	parallel = Parallel(n_jobs=n_jobs, verbose=0, backend="loky")
	tfs, errs = zip(*parallel(delayed(icp.icp)(
		np.c_[lidar_points[i],np.ones(len(lidar_points[i]))],
		np.c_[lidar_points[i-1],np.ones(len(lidar_points[i-1]))],
		init_transform=utils.pose_to_mat(pose_graph.poses[i] - pose_graph.poses[i-1]),
		max_iters=icp_max_iters,
		epsilon=icp_epsilon
	) for i in tqdm(range(1, len(pose_graph.poses)))))

	for i in range(len(pose_graph.poses)-1, 1-1, -1):
	# for i in range(1, len(pose_graph.poses)):
		real_tf = tfs[i-1][-1]
		dtheta = np.arctan2(real_tf[1][0], real_tf[0][0])
		pose_graph.poses[i][2] = pose_graph.poses[i-1][2]

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

