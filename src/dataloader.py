import cv2
import numpy as np
import lcm
import sys
from tqdm import tqdm
import os
sys.path.append(sys.path[0] + "/lcmtypes")
try:
	from src.lcmtypes import odometry_t, lidar_t
except:
	from lcmtypes import odometry_t, lidar_t


def read_img(path):
	image = cv2.imread(path, cv2.COLOR_BGR2RGB)
	return image


def get_images(data_folder_name, image_stop):
	timestamps_file = open(f'{data_folder_name}/image_timestamps.txt', 'r')
	lines = timestamps_file.readlines()
	imgs = np.empty((0, 480, 640, 3), dtype=float)
	timestamps = np.empty(0, dtype=float)
	
	print("Loading images...")
	if image_stop > len(lines):
		image_stop = len(lines)
	for i in tqdm(range(0, image_stop+1)):
		line = lines[i]
		n, time = line.split(", ")
		if int(n) > image_stop:
			break
		img = read_img(f'{data_folder_name}/raw_images/image{n}.png')
		imgs = np.append(imgs, np.array([img]), axis=0)
		timestamps = np.append(timestamps, float(time))

	timestamps *= 1E6

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
	odometry = np.empty((0, 3), dtype=float)
	odometry_timestamps = np.empty(0, dtype=float)
	point_cloud = []
	point_cloud_timestamps = np.empty(0, dtype=float)

	log_fname = ""
	for file in os.listdir(data_folder_name):
		if file.endswith(".log"):
			log_fname = os.path.join(data_folder_name, file)
	log = lcm.EventLog(log_fname, "r")
	
	for event in log:
		if event.channel == "ODOMETRY":
			msg = odometry_t.decode(event.data)
			odometry = np.append(odometry, np.array([[msg.x, msg.y, msg.theta]]), axis=0)
			odometry_timestamps = np.append(odometry_timestamps, msg.utime)
		if event.channel == "LIDAR":
			msg = lidar_t.decode(event.data)
			point_cloud.append(get_point_cloud(msg.ranges, msg.thetas))
			point_cloud_timestamps = np.append(point_cloud_timestamps, msg.utime)
	return odometry, odometry_timestamps, point_cloud, point_cloud_timestamps


def align_data(odometry, odometry_timestamps, point_clouds, point_cloud_timestamps, images, image_timestamps):
	if images is not None:
		final_odometry = np.empty((0, 3), dtype=float)
		final_point_cloud = []
		for i in range(images.shape[0]):
			time = image_timestamps[i]
			odo_idx = np.searchsorted(odometry_timestamps, time)
			point_idx = np.searchsorted(point_cloud_timestamps, time)
			odometry_to_add = odometry[odo_idx if odo_idx < odometry.shape[0] else -1].reshape((1, 3))
			final_odometry = np.append(final_odometry, odometry_to_add, axis=0)
			point_cloud_to_add = point_clouds[point_idx if point_idx < len(point_clouds) else -1]
			final_point_cloud.append(point_cloud_to_add)
		return final_odometry, final_point_cloud, images
	else:
		final_odometry = np.empty((0, 3), dtype=float)
		for i in range(len(point_clouds)):
			time = point_cloud_timestamps[i]
			odo_idx = np.searchsorted(odometry_timestamps, time)
			odometry_to_add = odometry[odo_idx if odo_idx < odometry.shape[0] else -1].reshape((1, 3))
			final_odometry = np.append(final_odometry, odometry_to_add, axis=0)
		return final_odometry, point_clouds


def parse_lcm_log(data_folder_name, start_time=0, stop_time=np.inf, load_images=True, image_stop=np.inf):
	# This script should read in the log file and images, matches lidar and odometry data
	# to each image, and then returns three things:
	# (1) An (n, 3) numpy array of odometry values (x, y, theta)
	# (2) A length-n array of (m_i, 2) point cloud lidar data -- each m_i-by-2 entry of the length-n
	#     list is in its own local coordinate frame (i.e. the robot center is at 0,0, and the
	#     forward direction is positive x)
	# (3) An (n, h, w, 3) numpy array of camera images
	# n is the number of camera images, m is the number of ranges returned by the LIDAR sensor,
	# and w, h are the dimensions of the image

	# TODO: Use start and stop time

	odometry, odometry_timestamps, point_clouds, point_cloud_timestamps = get_all_lcm_data(data_folder_name)
	if load_images:
		images, image_timestamps = get_images(data_folder_name, image_stop=image_stop)
	else:
		images, image_timestamps = None, None
		
	return align_data(odometry, odometry_timestamps, point_clouds, point_cloud_timestamps, images, image_timestamps)



def run_test():
	odometry, point_clouds, images = parse_lcm_log("./data/lab_maze")
	print(odometry.shape)
	print(point_clouds.shape)
	print(images.shape)


if __name__ == '__main__':
	run_test()


