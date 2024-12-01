#!/usr/bin/env python3

import cv2
from cv_bridge import CvBridge

import requests
import json_numpy
json_numpy.patch()
import numpy as np
import time

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
        super().__init__("openvlaclient_dummy_node")
        # self.cap = cv2.VideoCapture(6)  # Open the default camera
        # self.bridge = CvBridge()  # ROS2 to OpenCV converter
        self.action_topic = "/openvla_action_dummy"
        self.action_puber = self.create_publisher(Float32MultiArray, self.action_topic, 10)
        self.processing_timer = self.create_timer(2.0, self.action_pub)

        # Create a lightweight 256x256 grayscale (single-channel) dummy image
        resized_dummy_image = np.full((256, 256), 150, dtype=np.uint8) 
        self.image = resized_dummy_image

        self.instruction = instruction if instruction else "pick up banana"
        self.unnorm_key = 'berkeley_autolab_ur5'
        self.count = 0
            

    def action_pub(self):
        # Record the start time
        start_time = time.time()
        print("--------------------")
        print("start inference")

        print("unnormalised with ", self.unnorm_key)

        # Send request to server
        response = requests.post(
            "http://0.0.0.0:8000/act",
            json={"image": self.image, "instruction": self.instruction, "unnorm_key": self.unnorm_key}
        ).json()

        # Record the end time
        end_time = time.time()
        # Calculate the elapsed time in seconds
        elapsed_time_minutes = (end_time - start_time)

        action = response.tolist()
        # print("Action predicted by the model:", action)
        print(f"It took {elapsed_time_minutes:.4f} seconds for inference")

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
