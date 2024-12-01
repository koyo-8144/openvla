import requests
import json_numpy
json_numpy.patch()
import numpy as np
import time


## example ##
# action = requests.post(
#     "http://0.0.0.0:8000/act",
#     json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
# ).json()


# Example image as a NumPy array
image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)  # Replace with actual image data
print("image: random image")

# Example instruction
instruction = "do something"
print("instruction: ", instruction)

# Add a valid unnorm_key from the list of available keys
unnorm_key = 'berkeley_autolab_ur5'
"""
'austin_buds_dataset_converted_externally_to_rlds', 'austin_sailor_dataset_converted_externally_to_rlds', 'austin_sirius_dataset_converted_externally_to_rlds', 'bc_z', 'berkeley_autolab_ur5', 
'berkeley_cable_routing', 'berkeley_fanuc_manipulation', 'bridge_orig', 'cmu_stretch', 'dlr_edan_shared_control_converted_externally_to_rlds', 'dobbe', 'fmb_dataset', 'fractal20220817_data', 
'furniture_bench_dataset_converted_externally_to_rlds', 'iamlab_cmu_pickup_insert_converted_externally_to_rlds', 'jaco_play', 'kuka', 'nyu_franka_play_dataset_converted_externally_to_rlds', 'roboturk', 
'stanford_hydra_dataset_converted_externally_to_rlds', 'taco_play', 'toto', 'ucsd_kitchen_dataset_converted_externally_to_rlds', 'utaustin_mutex', 'viola'
"""

# Record the start time
# start_time = time.time()
print("--------------------")
print("start inference")

# Send request to server
response = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": image, "instruction": instruction, "unnorm_key": unnorm_key}
).json()

action = response.tolist()

# Record the end time
# end_time = time.time()
# Calculate the elapsed time in seconds
# elapsed_time_minutes = (end_time - start_time)

print("Action predicted by the model:", action)
# print(f"It took {elapsed_time_minutes:.4f} seconds for inference")

# print(f"{len(action)} DOF")


# # List of available unnorm_keys
# unnorm_keys = [
#     'austin_buds_dataset_converted_externally_to_rlds', 'austin_sailor_dataset_converted_externally_to_rlds',
#     'austin_sirius_dataset_converted_externally_to_rlds', 'bc_z', 'berkeley_autolab_ur5', 
#     'berkeley_cable_routing', 'berkeley_fanuc_manipulation', 'bridge_orig', 'cmu_stretch', 
#     'dlr_edan_shared_control_converted_externally_to_rlds', 'dobbe', 'fmb_dataset', 'fractal20220817_data', 
#     'furniture_bench_dataset_converted_externally_to_rlds', 'iamlab_cmu_pickup_insert_converted_externally_to_rlds', 
#     'jaco_play', 'kuka', 'nyu_franka_play_dataset_converted_externally_to_rlds', 'roboturk', 
#     'stanford_hydra_dataset_converted_externally_to_rlds', 'taco_play', 'toto', 
#     'ucsd_kitchen_dataset_converted_externally_to_rlds', 'utaustin_mutex', 'viola'
# ]

# # Iterate over all unnorm_keys
# for unnorm_key in unnorm_keys:
#     print(f"Testing unnorm_key: {unnorm_key}")

#     print("--------------------")
#     print("start inference")

#     # Send request to server
#     response = requests.post(
#         "http://0.0.0.0:8000/act",
#         json={"image": image, "instruction": instruction, "unnorm_key": unnorm_key}
#     ).json()

#     action = response.tolist()

#     print(f"Action predicted by the model: {action}")
#     print(f"{len(action)} DOF")
#     print("--------------------")
