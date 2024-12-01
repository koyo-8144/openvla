import requests
import json_numpy
json_numpy.patch()
import numpy as np
import time


#!/usr/bin/env python3
import argparse
from PIL import Image
import numpy as np
from io import BytesIO
from sensor_msgs.msg import Image as ROS2Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float32MultiArray
import rclpy
from rclpy.node import Node

from queue import Queue
from threading import Thread
import cv2
from cv_bridge import CvBridge
import sys 
import ast
from PIL import ImageDraw, ImageFont

import requests
import json_numpy
json_numpy.patch()
import numpy as np
import time



class OpenVLAClient_ImageSub(Node):
    def __init__(self, instruction=None):
        super().__init__("openvlaclient_node")
        # device, dtype = detect_device()
        # # device = torch.device("cpu")
        # # dtype = torch.float32

        # # Subscribers
        # self.image_topic = "/wrist_mounted_camera/image"
        # self.image_suber = self.create_subscription(ROS2Image, self.image_topic, self.image_callback, 10)
        # self.image_data = None
        # self.latest_image_embeds = None
        # # self.bridge = CvBridge()  # ROS2 to OpenCV converter

        # # Minimum interval between callbacks in seconds
        # self.min_interval = 2.0  # Adjust this as needed
        # self.last_callback_time = time.time()  # Initialize with the current time

        # Publisher
        self.action_topic = "/openvla_action"
        self.action_puber = self.create_publisher(Float32MultiArray, self.action_topic, 10)
        self.processing_timer = self.create_timer(2.0, self.action_pub)


        # Example image as a NumPy array
        self.image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)  # Replace with actual image data
        

        if instruction == None:
            self.instruction = "do something"
        else:
            self.instruction = instruction

        # Add a valid unnorm_key from the list of available keys
        self.unnorm_key = 'berkeley_autolab_ur5'
        """
        'austin_buds_dataset_converted_externally_to_rlds', 'austin_sailor_dataset_converted_externally_to_rlds', 'austin_sirius_dataset_converted_externally_to_rlds', 'bc_z', 'berkeley_autolab_ur5', 
        'berkeley_cable_routing', 'berkeley_fanuc_manipulation', 'bridge_orig', 'cmu_stretch', 'dlr_edan_shared_control_converted_externally_to_rlds', 'dobbe', 'fmb_dataset', 'fractal20220817_data', 
        'furniture_bench_dataset_converted_externally_to_rlds', 'iamlab_cmu_pickup_insert_converted_externally_to_rlds', 'jaco_play', 'kuka', 'nyu_franka_play_dataset_converted_externally_to_rlds', 'roboturk', 
        'stanford_hydra_dataset_converted_externally_to_rlds', 'taco_play', 'toto', 'ucsd_kitchen_dataset_converted_externally_to_rlds', 'utaustin_mutex', 'viola'
        """

        self.count = 0

        # self.action = [0] * 7


    # def image_callback(self, msg):
    #     # Check if the minimum interval has passed
    #     current_time = time.time()
    #     if current_time - self.last_callback_time < self.min_interval:
    #         return  # Skip this callback if the interval hasn't passed

    #     # Update the last callback time
    #     self.last_callback_time = current_time

    #     # Convert ROS2 image data to a NumPy array
    #     image = np.array(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
    #     self.get_logger().info("Subscribing image")
    #     # # Convert NumPy array to PIL image
    #     # self.image_data = Image.fromarray(img_np)
    #     # # print("image ", self.image_data) #<PIL.Image.Image image mode=RGB size=1280x720 at 0x752C9BBBBA00>

    #     # # Convert ROS2 image message to OpenCV format
    #     # cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    #     # # Display the image using OpenCV
    #     # cv2.imshow('Robot Camera View', cv_image)
    #     # # Add waitKey to allow image to render and enable keyboard control (exit with 'q')
    #     # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     #     rclpy.shutdown()


    #     # Record the start time
    #     # start_time = time.time()
    #     print("--------------------")
    #     print("start inference")

    #     # Send request to server
    #     action = requests.post(
    #         "http://0.0.0.0:8000/act",
    #         json={"image": image, "instruction": self.instruction, "unnorm_key": self.unnorm_key}
    #     ).json()

    #     self.action = action.tolist()

    #     # # Record the end time
    #     # end_time = time.time()
    #     # # Calculate the elapsed time in seconds
    #     # elapsed_time_minutes = (end_time - start_time)

    #     print("Action predicted by the model:", action)
    #     # print(f"It took {elapsed_time_minutes:.4f} seconds for inference")
    
    def action_pub(self,):
        # Record the start time
        # start_time = time.time()
        print("--------------------")
        print("start inference")

        # Send request to server
        response = requests.post(
            "http://0.0.0.0:8000/act",
            json={"image": self.image, "instruction": self.instruction, "unnorm_key": self.unnorm_key}
        ).json()

        action = response.tolist()

        # Record the end time
        # end_time = time.time()
        # Calculate the elapsed time in seconds
        # elapsed_time_minutes = (end_time - start_time)

        print("Action predicted by the model:", action)

        self.count += 1
        self.get_logger().info(f"Publishing openvla action, {action}, #{self.count}")
        msg = Float32MultiArray()
        msg.data = action
        self.action_puber.publish(msg)
        


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", type=str, required=False)
    args = parser.parse_args()

    rclpy.init(args=sys.argv)
    client_node = OpenVLAClient_ImageSub(instruction=args.instruction)
    rclpy.spin(client_node)       # Keeps the node running, processing incoming messages
    rclpy.shutdown()


if __name__ == '__main__':
    main()

