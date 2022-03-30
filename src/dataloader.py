import cv2
import numpy as np
import lcm
from lcmtypes import odometry_t, lidar_t


def read_img(path):
	image = cv2.imread(path, cv2.COLOR_BGR2RGB)
	return image


def get_images(data_folder_name):
	timestamps_file = open(f'{data_folder_name}/image_timestamps.txt', 'r')
	lines = timestamps_file.readlies()
	imgs = np.empty((0, 480, 640, 3), dtype=float)
	timestamps = np.empty((0), dtype=float)
	
	for line in lines:
		n, time = line.split(", ")
		img = read_img(f'{data_folder_name}/raw_images/image{n}.png')
		imgs = np.append(imgs, img)
		timestamps = np.append(timestamps, time)
		
	timestamps -= timestamps[0]

	return imgs, timestamps


def get_all_lcm_data(data_folder_name):
	odometry = np.empty((0, 3), dtype=float)
	point_cloud = np.empty((0, 0, 2), dtype=float)
	init = False
	
	log = lcm.EventLog(f'{data_folder_name}/lcm.log', "r")
	
	for event in log:
		if event.channel == "ODOMETRY":
			msg = odometry_t.decode(event.data)
			if not init:
				start_time = msg.utime
				init = True
			odometry = np.append(odometry, np.array([msg.utime - start_time, msg.x, msg.y, msg.theta], axis = 0)	
		

def parse_lcm_log(data_folder_name, start_time=0, stop_time=np.inf, load_images=True):
	# This script should read in the log file and images, matches lidar and odometry data
	# to each image, and then returns three things:
	# (1) An (n, 3) numpy array of odometry values (x, y, theta)
	# (2) A length-n array of (m_i, 2) point cloud lidar data -- each m_i-by-2 entry of the length-n
	#     list is in its own local coordinate frame (i.e. the robot center is at 0,0, and the
	#     forward direction is positive x)
	# (3) An (n, h, w, 3) numpy array of camera images
	# n is the number of camera images, m is the number of ranges returned by the LIDAR sensor,
	# and w, h are the dimensions of the image
	pass