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
import src.dataloader as dataloader

odometry, lidar_points = dataloader.parse_lcm_log("./data/lab_maze", load_images=False)

start = 10
cell_width = 0.05
og, (min_x, min_y) = produce_occupancy_grid.produce_occupancy_grid(odometry[:start], lidar_points[:start], cell_width, min_width=10, min_height=10)

fig, ax = plt.subplots()
visualization.draw_occupancy_grid(ax, og, cell_size=0.05, origin_location=np.array([0, 0]))
plt.draw()
plt.pause(0.001)

skip = 5
for i in range(start, len(odometry), skip):
	og = produce_occupancy_grid.update_occupancy_grid(og, odometry[i:i+skip], lidar_points[i:i+skip], cell_width, min_x, min_y)
	visualization.draw_occupancy_grid(ax, og, cell_size=0.05, origin_location=np.array([0, 0]))
	plt.draw()
	plt.pause(0.001)
