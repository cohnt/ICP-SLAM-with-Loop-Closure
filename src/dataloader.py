import cv2
import numpy as np
import lcm
import sys
import os
sys.path.append(sys.path[0] + "/lcmtypes")
try:
	from src.lcmtypes import odometry_t, lidar_t
except:
	from lcmtypes import odometry_t, lidar_t


def read_img(path):
	image = cv2.imread(path, cv2.COLOR_BGR2RGB)
	return image


def get_images(data_folder_name):
	timestamps_file = open(f'{data_folder_name}/image_timestamps.txt', 'r')
	lines = timestamps_file.readlines()
	imgs = np.empty((0, 480, 640, 3), dtype=float)
	timestamps = np.empty(0, dtype=float)
	
	for line in lines:
		n, time = line.split(", ")
		img = read_img(f'{data_folder_name}/raw_images/image{n}.png')
		imgs = np.append(imgs, np.array([img]), axis=0)
		timestamps = np.append(timestamps, time)

	return imgs, timestamps


def get_point_cloud(ranges, thetas):
	np_ranges = np.array(ranges).reshape((-1, 1))
	np_thetas = -np.array(thetas).reshape((-1, 1))
	valid = 0.05 < np_ranges  # RP Lidar advertises a 12 meter range
	np_ranges = np_ranges[valid]
	np_thetas = np_thetas[valid]
	x = np_ranges * np.cos(np_thetas)
	y = np_ranges * np.sin(np_thetas)
	return np.hstack((x.reshape((-1, 1)), y.reshape((-1, 1))))


def get_all_lcm_data(data_folder_name):
	odometry = np.empty((0, 4), dtype=float)
	point_cloud = []
	point_cloud_matched_idx = []

	log_fname = ""
	for file in os.listdir(data_folder_name):
		if file.endswith(".log"):
			log_fname = os.path.join(data_folder_name, file)
	log = lcm.EventLog(log_fname, "r")
	
	for event in log:
		if event.channel == "ODOMETRY":
			msg = odometry_t.decode(event.data)
			odometry = np.append(odometry, np.array([[msg.utime, msg.x, msg.y, msg.theta]]), axis=0)
		if event.channel == "LIDAR":
			msg = lidar_t.decode(event.data)
			point_cloud.append(get_point_cloud(msg.ranges, msg.thetas))
			point_cloud_matched_idx.append(len(odometry)-1)

	return odometry[point_cloud_matched_idx], point_cloud
		

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

	odometry, point_clouds = get_all_lcm_data(data_folder_name)
	print(odometry.shape)
	print(len(point_clouds))
	print(point_clouds[0].shape)

	if not load_images:
		return odometry[:,1:], point_clouds

	imgs, timestamps = get_images(data_folder_name)
	print(imgs.shape)
	print(timestamps.shape)



def run_test():
	parse_lcm_log("./data/lab_maze")


if __name__ == '__main__':
	run_test()


