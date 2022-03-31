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


fig, ax = plt.subplots()

for i in range(10, len(odometry), 10):
	og = produce_occupancy_grid.produce_occupancy_grid(odometry[:i], lidar_points[:i], 0.05, kHitOdds=100, kMissOdds=50)
	visualization.draw_occupancy_grid(ax, og, cell_size=0.05, origin_location=np.array([0, 0]))
	plt.draw()
	plt.pause(0.001)