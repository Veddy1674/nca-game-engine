import os
from glob import glob
import numpy as np

files = glob("snake/dev/data/snake_*.npz")

disk_gb = sum(os.path.getsize(f) for f in files) / (1024**3)
print(f"Disk space: {disk_gb:.3f} GB")

with np.load(files[0]) as data:
    state = data['states']
    action = data['actions']
    
    steps_per_file = len(state)
    state_bytes = state.nbytes
    action_bytes = action.nbytes
    
    file_gb = (state_bytes + action_bytes) / (1024**3)

print(f"Steps per file: {steps_per_file}")
print(f"RAM space: {file_gb:.2f} GB per file (Total: {len(files) * file_gb:.3f} GB)")

print(f"\nState shape: {state.shape}, Action shape: {action.shape}")