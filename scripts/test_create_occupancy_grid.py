#######################
# Fix import issues   #
import sys            #
sys.path.append(".")  #
sys.path.append("..") #
#######################

import numpy as np
import matplotlib.pyplot as plt

import src.produce_occupancy_grid as produce_occupancy_grid
import src.utils as utils
import src.visualization as visualization

poses = np.array([
	[0, 0, 0],
	[0, 1, np.pi/2]
])
ranges = np.array([
	[0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75],
	[1, 1, 1, 1, 0.5, 2, 2, 2]
])
angle_min = 0
angle_step = 2 * np.pi / ranges.shape[1]
lidar_points = np.array([
	[
		ranges[i,j] * np.array([np.cos(j*angle_step), np.sin(j*angle_step)]) for j in range(ranges.shape[1])
	] for i in range(ranges.shape[0])
])

og = produce_occupancy_grid.produce_occupancy_grid(poses, lidar_points, 0.05, kHitOdds=100, kMissOdds=50)

fig, ax = plt.subplots()
visualization.draw_occupancy_grid(ax, og, cell_size=0.05, origin_location=np.array([0, 0]))
plt.show()

fig, ax = plt.subplots()
visualization.draw_occupancy_grid(ax, produce_occupancy_grid.grid_mle(og, unknown_empty=True), cell_size=0.05, origin_location=np.array([0, 0]))
plt.show()
fig, ax = plt.subplots()
visualization.draw_occupancy_grid(ax, produce_occupancy_grid.grid_mle(og, unknown_empty=False), cell_size=0.05, origin_location=np.array([0, 0]))
plt.show()

# produce_occupancy_grid.save_grid(og, "test_grid.map", 0.05)