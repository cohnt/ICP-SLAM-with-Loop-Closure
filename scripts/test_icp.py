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

list_of_transforms = icp.icp(pc1.T, pc2.T)
# print(list_of_transforms)

fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.scatter(pc1[0,:], pc1[1,:], color="red")
ax.scatter(pc2[0,:], pc2[1,:], color="blue")
plt.draw()
plt.pause(0.001)
for tf in list_of_transforms:
	print(tf)
	temp = tf @ pc1
	# corresp = icp.correspondences(temp, pc2)
	visualization.draw_icp_iteration(ax, temp.T, pc2.T)
	plt.draw()
	plt.pause(0.5)
plt.show()