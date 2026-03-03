import numpy as np
from env import SimpleColorEnv
from time import time

# save sand_0.npz
env = SimpleColorEnv(use_rgb=True)

states = []

state = env.reset()

t = time()
for i in range(400):
    states.append(state)

    state = env.step()

# fake actions
actions = np.array([0] * len(states))
np.savez_compressed('colorchange/dev/data/colorchangeRGB_0.npz', states=np.array(states), actions=actions)

print(f"Done in {time() - t:.2f}s.")