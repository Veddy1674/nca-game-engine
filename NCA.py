import torch
import torch.nn as nn
import torch.nn.functional as F

from glob import glob
import numpy as np

class NCA(nn.Module):
    """
    A Neural Cellular Automata PyTorch implementation specifically meant for Neural Game Engines.
    
    Args:
        `actions`: Number of action channels that each cell can perceive.
        `vis_channels`: Number of one-hot visible channels, the last one is the "dead" color (e.g. 4 for orange, red, purple, white, where white is alpha/background).
        `grid_size`: Size of the game grid (H, W).
        `hid_channels`: Number of hidden channels, that each cell can use for internal states.
        `input_length`: How many past states the cell can see (default: 1, meaning only the current state).
        `device`: Device to run the model on (default: cuda if available, else cpu).
    """
    def __init__(self, actions: int, vis_channels: int, hid_channels: int, hidden_neurons=128, input_length=1, device: str=None):
        super().__init__()

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device

        self.actions = actions # can be 0 or >= 2, '1' makes no sense
        
        self.input_length = input_length

        self.vis_channels = vis_channels
        self.hid_channels = hid_channels # causes issues if '1', 0 or >= 2 is fine
        self.channels = vis_channels + hid_channels

        self.input_dim = ((self.channels * 5) * self.input_length) + actions # cell sees itself (id) and up down left right (raw)

        if hid_channels > 0 and (hidden_neurons / hid_channels) <= 2:
            print("Warning: 'hidden_neurons / hid_channels' is less than or equal to 2, this might cause instability.")

        # update net
        self.net = nn.Sequential(
            # hidden neurons / hidden channels > 2
            nn.Conv2d(self.input_dim, hidden_neurons, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_neurons, self.channels, kernel_size=1), # delta!
            nn.SELU() # SELU() grants a faster training (lower loss), but a little more unstable than Tanh()
        )

        # init to zero the second Conv2d for stability
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

        kernels = torch.zeros(5, 3, 3)

        kernels[0, 1, 1] = 1.0 # center (id)
        kernels[1, 0, 1] = 1.0 # up
        kernels[2, 2, 1] = 1.0 # down
        kernels[3, 1, 0] = 1.0 # left
        kernels[4, 1, 2] = 1.0 # right

        # (60, 1, 3, 3)
        self.register_buffer('vn_kernel', kernels.unsqueeze(1).repeat(self.channels, 1, 1, 1))
        
        print(f"Using {device.upper()}")
        self.to(device)
    
    # Von Neumann neighborhood
    def perceive(self, x: torch.Tensor):
        # x shape is BCHW
        return F.conv2d(x, self.vn_kernel, groups=self.channels, padding=1) # border sees nothing (0.0)
    
    """ where input_length == 1:
    def forward(self, x: torch.Tensor, action_map: torch.Tensor):
        percept = self.perceive(x)

        # B, Actions, H, W
        inp = torch.cat([percept, action_map], dim=1)

        # calc delta
        dx = self.net(inp)

        return x + dx
    """

    # where input_length >= 1:
    def forward(self, states: list[torch.Tensor], action_map: torch.Tensor):
        # perceive each state and concat
        percepts = [self.perceive(s) for s in states]

        if action_map is not None:
            inp = torch.cat([*percepts, action_map], dim=1)
        else:
            inp = torch.cat(percepts, dim=1)
        
        dx = self.net(inp)
        return states[0] + dx # apply delta to current state only
    
    # s + a = s'
    """ where input_length == 1:
    def step(self, x: torch.Tensor, action_map: torch.Tensor, microsteps: int):
        state = x

        for _ in range(microsteps):
            state = self(state, action_map)
            
        return state
    """

    # e.g: (with input_length = 3): s + s' + s'' + a = s'''
    # where input_length >= 1
    def step(self, states: list[torch.Tensor], action_map: torch.Tensor, microsteps: int):
        state_history = states  # [current, prev1, prev2, ...]
        
        for _ in range(microsteps):
            new_state = self(state_history, action_map)
            # shift history: new becomes current, drop oldest
            state_history = [new_state] + state_history[:-1]
            
        return state_history[0]
    
    def save(self, file_name: str, optimizer: torch.optim.Optimizer):
        torch.save({
            'model_sd': self.state_dict(),
            'optimizer_sd': optimizer.state_dict()
        }, file_name)

    def load(self, file_name: str, optimizer: torch.optim.Optimizer|None=None):
        sav = torch.load(file_name, map_location=self.device)

        self.load_state_dict(sav['model_sd'])

        if optimizer is not None and 'optimizer_sd' in sav:
            optimizer.load_state_dict(sav['optimizer_sd'])

    @staticmethod
    def load_data(glob_files: str):
        # load all data to RAM (more efficient than a DataLoader for small datasets)
        files = glob(glob_files)
        
        states = [torch.from_numpy(np.load(f)['states']).float() for f in files]
        actions = [torch.from_numpy(np.load(f)['actions']).long() for f in files]

        return states, actions

    # loads the state at file 'file', at index 'idx', and returns as ndarray, useful for interference
    @staticmethod
    def load_data_first(filename: str, idx: int=0):
        states = np.load(filename)['states']
        return states[idx] # as tensor: torch.from_numpy(states[idx]).float()