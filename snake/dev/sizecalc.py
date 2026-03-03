# calculate the estimated RAM usage of the saved datasets
import numpy as np
from glob import glob

all_files = glob("data/snake_*.npz")
states_size = 0
actions_size = 0

for f in all_files:
    with np.load(f) as data:
        states_size += data['states'].size
        actions_size += data['actions'].size

# gb calc (DURING training, not the data itself, which is compressed)
size_states = (states_size * 4) / (1024**3)
size_actions = (actions_size * 8) / (1024**3)

print(f"States:  {size_states:.4f} GB - {size_states*1024:.2f} MB")
print(f"Actions: {size_actions:.4f} GB - {size_actions*1024:.2f} MB")
print(f"Total:  {size_states + size_actions:.4f} GB - {(size_states + size_actions)*1024:.2f} MB")
    