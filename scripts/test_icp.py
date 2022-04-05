#######################
# Fix import issues   #
import sys            #
sys.path.append(".")  #
sys.path.append("..") #
#######################

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import special_ortho_group

import src.visualization as visualization
import src.icp as icp

pc1 = (np.random.random(size=(20,2)) - 0.5) * 20
pc1 = np.hstack((pc1, np.ones((len(pc1),1)))).T

translation = np.random.random(size=(1,2))
# rotation = special_ortho_group.rvs(2)
theta = (np.random.random() - 0.5) * 0.1
c = np.cos(theta)
s = np.sin(theta)
rotation = np.array([
	[c, -s],
	[s, c]
])

transformation_mat = np.eye(3)
transformation_mat[:2,:2] = rotation
transformation_mat[:2,2] = translation

pc2 = transformation_mat @ pc1

# fig, ax = plt.subplots()
# ax.set_aspect("equal")
# ax.scatter(pc1[0,:], pc1[1,:], color="red")
# ax.scatter(pc2[0,:], pc2[1,:], color="blue")
# plt.show()

# list_of_transforms = icp.icp(pc1.T, pc2.T)
# print(list_of_transforms)

transforms = [np.eye(3)]

fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.scatter(pc1[0,:], pc1[1,:], color="red")
ax.scatter(pc2[0,:], pc2[1,:], color="blue")
plt.draw()
plt.pause(0.001)
for i in range(100):
	next_transform, correspondences, error = icp.icp_iteration(pc1.T, pc2.T, transforms[-1])
	ax.cla()
	print(next_transform)
	temp = next_transform @ pc1
	transforms.append(next_transform)
	visualization.draw_icp_iteration(ax, temp.T, pc2.T, correspondences)
	plt.draw()
	print(error)
	if error < 0.01:
		break
	plt.pause(1)

plt.show()