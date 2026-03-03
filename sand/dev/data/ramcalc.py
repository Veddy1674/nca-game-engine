import os
from glob import glob
import numpy as np

files = glob("sand/dev/data/sand_*.npz")

disk_gb = sum(os.path.getsize(f) for f in files) / (1024**3)
print(f"Disk space: {disk_gb:.3f} GB")

with np.load(files[0]) as data:
    n_steps = len(data[data.files[0]]) # get steps

    state = data['states']
    action = data['actions']
    
    state_bytes = np.prod(state.shape[1:]) * state.itemsize
    action_bytes = action.itemsize

ram_gb = n_steps * (state_bytes + action_bytes) / (1024**3)
print(f"RAM space: {ram_gb:.2f} GB per file (Total: {len(files) * ram_gb:.3f} GB)")
print(f"\nState shape: {state.shape}, Action shape: {action.shape}")