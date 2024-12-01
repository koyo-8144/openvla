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
        super().__init__("openvlaclient_node")
        self.cap = cv2.VideoCapture(6)  # Open the default camera  #v4l2-ctl --list-devices
        self.bridge = CvBridge()  # ROS2 to OpenCV converter
        self.action_topic = "/openvla_action"
        self.action_puber = self.create_publisher(Float32MultiArray, self.action_topic, 10)
        self.processing_timer = self.create_timer(2.0, self.action_pub)
        self.image = None  # Placeholder for the captured image
        self.instruction = instruction if instruction else "pick up a ball"

        # self.unnorm_key = 'berkeley_autolab_ur5'                                   # Mean Translation:  0.0026666666070620203, Mean Rotation:  0.007294989312737207
        # abdc
        # self.unnorm_key = 'austin_buds_dataset_converted_externally_to_rlds'       # Mean Translation:  0.08395145640653723,   Mean Rotation:  0.0
        # asdc
        self.unnorm_key = 'austin_sailor_dataset_converted_externally_to_rlds'       # Mean Translation:  0.1359757178749134,    Mean Rotation:  0.00796218450162925
        # self.unnorm_key = 'austin_sirius_dataset_converted_externally_to_rlds'     # Mean Translation:  0.13550121903029919,   Mean Rotation:  -0.012113924846929672
        # self.unnorm_key = 'bc_z'                                                   # Mean Translation:  0.0025140739171918094, Mean Rotation:  0.010301201883679145
        # self.unnorm_key = 'berkeley_autolab_ur5'                                   # Mean Translation:  0.0007843137079594174, Mean Rotation:  0.004538188673848028
        # self.unnorm_key = 'berkeley_cable_routing'                                 # Mean Translation:  0.03816568030192985,   Mean Rotation:  0.08733366843415243
        # self.unnorm_key = 'berkeley_fanuc_manipulation'                            # Mean Translation:  0.000888888869020674,  Mean Rotation:  0.0005475542708939218
        # self.unnorm_key = 'bridge_orig'                                            # Mean Translation:  0.005204311384143757,  Mean Rotation:  0.00930391667150196
        # self.unnorm_key = 'cmu_stretch'                                            # Mean Translation:  0.0031634923837716732, Mean Rotation:  0.0
        # self.unnorm_key = 'dlr_edan_shared_control_converted_externally_to_rlds'   # Mean Translation:  0.017963196501890617,  Mean Rotation:  -0.010645498624692376
        # self.unnorm_key = 'dobbe'                                                  # Mean Translation:  0.002944041057794155,  Mean Rotation:  0.0024276331379352004
        # self.unnorm_key = 'fmb_dataset'                                            # Mean Translation:  -0.030896356955073274, Mean Rotation:  0.12934640762852692
        # self.unnorm_key = 'fractal20220817_data'                                   # Mean Translation:  0.0034235498613975437, Mean Rotation:  0.09794089281286265
        # self.unnorm_key = 'furniture_bench_dataset_converted_externally_to_rlds'   # Mean Translation:  0.009085779433449084,  Mean Rotation:  0.08420199344984051
        # self.unnorm_key = 'iamlab_cmu_pickup_insert_converted_externally_to_rlds'  # Mean Translation:  0.25442154960402474,   Mean Rotation:  0.2640279312731297
        # self.unnorm_key = 'jaco_play'                                              # Mean Translation:  0.016209150568332553,  Mean Rotation:  0.0
        # self.unnorm_key = 'kuka'                                                   # Mean Translation:  0.02345914795809716,   Mean Rotation:  -0.017556951102478013
        # self.unnorm_key = 'nyu_franka_play_dataset_converted_externally_to_rlds'   # Mean Translation:  0.004511542504714218,  Mean Rotation:  0.013635466178453062
        # self.unnorm_key = 'roboturk'                                               # Mean Translation:  0.030329989226815013,  Mean Rotation:  0.016178082586114004
        # self.unnorm_key = 'stanford_hydra_dataset_converted_externally_to_rlds'    # Mean Translation:  0.0027521689317243927, Mean Rotation:  0.04489585930050555
        # self.unnorm_key = 'taco_play'                                              # Mean Translation:  0.12472115811101747,   Mean Rotation:  -0.0014539842141999548
        # self.unnorm_key = 'toto'                                                   # Mean Translation:  0.3220751004867694,    Mean Rotation:  -0.07376155011170656
        # self.unnorm_key = 'ucsd_kitchen_dataset_converted_externally_to_rlds'      # Mean Translation:  276.6788542938731,     Mean Rotation:  17.012420222533294
        # self.unnorm_key = 'utaustin_mutex'                                         # Mean Translation:  0.1846741350842457,    Mean Rotation:  0.0017857130933431238
        # self.unnorm_key = 'viola'                                                  # Mean Translation:  0.13117459887772606,   Mean Rotation:  0.023588239007136403
        """
        'austin_buds_dataset_converted_externally_to_rlds', 'austin_sailor_dataset_converted_externally_to_rlds', 'austin_sirius_dataset_converted_externally_to_rlds', 'bc_z', 'berkeley_autolab_ur5', 
        'berkeley_cable_routing', 'berkeley_fanuc_manipulation', 'bridge_orig', 'cmu_stretch', 'dlr_edan_shared_control_converted_externally_to_rlds', 'dobbe', 'fmb_dataset', 'fractal20220817_data', 
        'furniture_bench_dataset_converted_externally_to_rlds', 'iamlab_cmu_pickup_insert_converted_externally_to_rlds', 'jaco_play', 'kuka', 'nyu_franka_play_dataset_converted_externally_to_rlds', 'roboturk', 
        'stanford_hydra_dataset_converted_externally_to_rlds', 'taco_play', 'toto', 'ucsd_kitchen_dataset_converted_externally_to_rlds', 'utaustin_mutex', 'viola'
        """

        # List of unnormalized keys
        self.unnorm_keys = [
            'austin_buds_dataset_converted_externally_to_rlds',
            'austin_sailor_dataset_converted_externally_to_rlds',
            'austin_sirius_dataset_converted_externally_to_rlds',
            'bc_z',
            'berkeley_autolab_ur5',
            'berkeley_cable_routing',
            'berkeley_fanuc_manipulation',
            'bridge_orig',
            'cmu_stretch',
            'dlr_edan_shared_control_converted_externally_to_rlds',
            'dobbe',
            'fmb_dataset',
            'fractal20220817_data',
            'furniture_bench_dataset_converted_externally_to_rlds',
            'iamlab_cmu_pickup_insert_converted_externally_to_rlds',
            'jaco_play',
            'kuka',
            'nyu_franka_play_dataset_converted_externally_to_rlds',
            'roboturk',
            'stanford_hydra_dataset_converted_externally_to_rlds',
            'taco_play',
            'toto',
            'ucsd_kitchen_dataset_converted_externally_to_rlds',
            'utaustin_mutex',
            'viola'
        ]
        self.current_key_index = 0  # Track the current key index
        # self.processing_timer = self.create_timer(2.0, self.process_key)

        self.translation_list = []
        self.rotation_list = []
        self.count = 0
    
    def process_key(self):
        if self.current_key_index < len(self.unnorm_keys):
            self.unnorm_key = self.unnorm_keys[self.current_key_index]
            self.action_pub()
            self.current_key_index += 1
        else:
            self.get_logger().info("Processed all unnormalized keys.")
            rclpy.shutdown()

    def capture_and_render_image(self):
        #while True:
        # Capture a frame from the camera
        ret, frame = self.cap.read() #frame, (1080, 1920)
        if ret:
            # Convert the frame to the expected format if necessary
            cropped_image = frame[0:1080,400:1920]
            resized_image = cv2.resize(cropped_image, (256, 256))
            self.image = resized_image
            # print("self.image ", self.image.shape[:2])
            # Render the image in a window
            # cv2.imshow("Cropped Image", cropped_image)
            cv2.imshow("Cropped Image", resized_image)
            # WaitKey allows image rendering and checks if 'q' was pressed to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rclpy.shutdown()
        else:
            self.get_logger().error("Failed to capture image")

    def action_pub(self):
        self.capture_and_render_image()  # Capture a new image before sending it
        if self.image is None:
            return  # Skip if no image is available

        # Record the start time
        start_time = time.time()
        print("--------------------")
        # print("Start inference")

        print("Unnormalised with", self.unnorm_key)

        # Send request to server
        response = requests.post(
            "http://0.0.0.0:8000/act",
            json={"image": self.image, "instruction": self.instruction, "unnorm_key": self.unnorm_key}
        ).json()
        print(response)

        # Record the end time
        end_time = time.time()
        # Calculate the elapsed time in seconds
        elapsed_time_minutes = (end_time - start_time)
        # print(f"It took {elapsed_time_minutes:.4f} seconds for inference")

        action = response.tolist()
        # translation = action[0:3]
        # rotation = action[3:6]
        # self.translation_list.append(translation)
        # self.rotation_list.append(rotation)

        # # Find maximum values in translation and rotation lists
        # if self.translation_list:
        #     max_translation = np.max(np.array(self.translation_list), axis=0)
        #     print(f"Maximum Translation: {max_translation}")
        #     print(f"Mean Translation: ", np.mean(max_translation))

        # if self.rotation_list:
        #     max_rotation = np.max(np.array(self.rotation_list), axis=0)
        #     print(f"Maximum Rotation: {max_rotation}")
        #     print(f"Mean Rotation: ", np.mean(max_rotation))

        self.count += 1
        self.get_logger().info(f"Publishing openvla action, {action}, #{self.count}")
        msg = Float32MultiArray()
        msg.data = action
        self.action_puber.publish(msg)

        print("--------------------")

        # if self.count == 10:
        #     print("Reached to count 10")
        #     breakpoint()

    def __del__(self):
        # Release the camera when the node is destroyed
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

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
