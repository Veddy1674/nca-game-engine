from glob import glob
import numpy as np
import torch

data = np.load('data/snake_0.npz')

state = torch.tensor(data['states']).long()
action = torch.tensor(data['actions']).long()

print(state.shape)
print(action.shape)

print(state[3])
print(state[4])