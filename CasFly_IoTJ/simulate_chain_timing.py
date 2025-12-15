"""
simulate_chain_timing.py

Simulates chains of 1 to 100 devices to measure how total wall time scales
with chain length in the CasFly framework.

For each chain length L and each trial, L device modules are dynamically
loaded (cycling over the 26 real device scripts as needed), then
device_mod.initiate_chain() is called sequentially on each device.
Wall-clock time is recorded per trial and averaged.

Inputs:
    data/synthea/device_scripts/Device{1..26}/Device{N}.py
        -- real device scripts that expose initiate_chain(event, patient_id)

Outputs:
    framework_chain_time.png -- line plot of mean total time vs chain length
"""

import time
import importlib.util
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Parameters
min_chain_length = 1
max_chain_length = 100   # simulate up to 100 devices
num_trials       = 10    # independent trials per chain length

num_real_devices = 26    # number of actual device scripts available


def get_device_module(device_num):
    """
    Load and return the Python module for the given (possibly wrapped) device number.
    If device_num exceeds num_real_devices, it wraps around the pool of real devices
    so we can simulate arbitrarily long chains with only 26 real scripts.
    """
    # Map logical device number to an actual script index (1-indexed, cyclic)
    device_num_in_pool = ((device_num - 1) % num_real_devices) + 1
    device_folder = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        f"../data/synthea/device_scripts/Device{device_num_in_pool}"
    ))
    device_py_path = os.path.join(device_folder, f"Device{device_num_in_pool}.py")

    # Ensure the device folder is in sys.path so any relative imports inside the script work
    if device_folder not in sys.path:
        sys.path.insert(0, device_folder)

    module_name = f"Device{device_num_in_pool}_module"
    spec   = importlib.util.spec_from_file_location(module_name, device_py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


chain_lengths       = list(range(min_chain_length, max_chain_length + 1))
mean_times_per_length = []  # one entry per chain length: average wall-clock time across trials

for length in chain_lengths:
    trial_times = []
    for _ in range(num_trials):
        # Load one module per device in this chain (wrapping around the real pool as needed)
        device_modules = [get_device_module(i + 1) for i in range(length)]
        event      = "simulated_event"
        start_time = time.time()
        for idx, device_mod in enumerate(device_modules):
            # Call the device's chain initiation function sequentially
            device_mod.initiate_chain(event, patient_id=f"sim_{idx}")
        end_time = time.time()
        trial_times.append(end_time - start_time)
    mean_times_per_length.append(sum(trial_times) / len(trial_times))

plt.figure(figsize=(8, 5))
plt.plot(chain_lengths, mean_times_per_length, marker='o')
plt.xlabel('Chain Length (Number of Devices)')
plt.ylabel('Total Time (seconds)')
plt.title('Framework: Total Time vs. Chain Length (Simulated Devices)')
plt.grid(True)
plt.tight_layout()
plt.savefig('framework_chain_time.png')
plt.show()

print('Simulation complete. Plot saved as framework_chain_time.png.')
