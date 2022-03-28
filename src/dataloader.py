import numpy as np

def parse_lcm_log(data_folder_name, start_time=0, stop_time=np.inf, load_images=True):
	# This script should read in the log file and images, matches lidar and odometry data
	# to each image, and then returns three things:
	# (1) An (n, 3) numpy array of odometry values (x, y, theta)
	# (2) An (n, m, 2) numpy array of point cloud lidar data -- each m-by-2 entry of the length-n
	#     list is in its own local coordinate frame (i.e. the robot center is at 0,0, and the
	#     forward direction is positive x)
	# (3) An (n, h, w, 3) numpy array of camera images
	# n is the number of camera images, m is the number of ranges returned by the LIDAR sensor,
	# and w, h are the dimensions of the image
	pass