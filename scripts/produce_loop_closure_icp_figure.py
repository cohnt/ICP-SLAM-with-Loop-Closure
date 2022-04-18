#######################
# Fix import issues   #
import sys            #
import traceback      #
sys.path.append(".")  #
sys.path.append("..") #
#######################

import numpy as np
import matplotlib.pyplot as plt
import src.dataloader as dataloader
import src.icp as icp
import src.visualization as visualization

print("Loading the data")
odometry, lidar_points, images = dataloader.parse_lcm_log("./data/EECS_3", load_images=True, image_stop=np.inf, n_jobs=-1)

pc1 = lidar_points[56]
pc1 = np.hstack((pc1, np.ones((len(pc1),1)))).T
pc2 = lidar_points[182]
pc2 = np.hstack((pc2, np.ones((len(pc2),1)))).T

transforms = [np.eye(3)]

fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.scatter(pc1[0,:], pc1[1,:], color="red")
ax.scatter(pc2[0,:], pc2[1,:], color="blue")
plt.savefig("results/iter%02d.png" % 0)
for i in range(30):
	next_transform, correspondences, error = icp.icp_iteration(pc1.T, pc2.T, transforms[-1])
	ax.cla()
	print(next_transform)
	temp = next_transform @ pc1
	transforms.append(next_transform)
	visualization.draw_icp_iteration(ax, temp.T, pc2.T, correspondences)
	plt.savefig("results/iter%02d.png" % (i+1))
	print(error)
	if error < 0.01:
		break
