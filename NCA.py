import torch
import torch.nn as nn
import torch.nn.functional as F

from glob import glob
import numpy as np

class NCA(nn.Module):
    """
    A Neural Cellular Automata PyTorch implementation specifically meant for Neural Game Engines.
    
    Args:
        `actions`: Number of action channels that each cell perceives.
        `vis_channels`: Number of one-hot visible channels, the last one is the "dead" color (e.g. 4 for orange, red, purple, white, where white is alpha/background).
        `hid_channels`: Number of hidden channels, that each cell can use for internal states.
        `extra_channels`: Number of extra channels that each cell perceives, just like actions.
        `hidden_neurons`: Number of hidden neurons of the update net, which updates cells.
        `input_length`: How many past states the cell can see (default: 1, meaning only the current state).
        `padding_mode`: Padding mode for the perceive function: 'reflect', 'circular', 'replicate' or the default 'zeros'.
        `use_global_context`: Whether to give all cells a general context of all other cells (for example, how much of each color is visible).
        `dilations`: List of dilations to use for the perceive function (default: [1], meaning only the immediate neighbors).
        `device`: Device to run the model on (default: cuda if available, else cpu).
    """
    def __init__(self, actions: int, vis_channels: int, hid_channels: int, extra_channels:int=0, hidden_neurons:int=128, input_length:int=1, padding_mode: str = 'zeros', use_global_context: bool = False, dilations: list = [1], device: str=None):
        super().__init__()

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device
        self.padding_mode = padding_mode
        self.input_length = input_length
        self.use_global_context = use_global_context
        self.dilations = dilations

        self.actions = actions # can be 0 or >= 2, '1' makes no sense

        self.vis_channels = vis_channels
        self.hid_channels = hid_channels # causes issues if '1', 0 or >= 2 is fine
        self.channels = vis_channels + hid_channels

        # cell sees itself (id) and up down left right (raw)
        self.input_dim = ((self.channels * 5 * len(dilations)) * self.input_length) + actions + extra_channels + (self.channels if use_global_context else 0)

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

        # (self.channels, 1, 3, 3)
        self.register_buffer('vn_kernel', kernels.unsqueeze(1).repeat(self.channels, 1, 1, 1))
        
        print(f"Using {device.upper()}")
        self.to(device)
    
    # Von Neumann neighborhood
    def perceive(self, x: torch.Tensor):
        # x shape is BCHW

        perceptions = []
        for d in self.dilations:
            if self.padding_mode == 'zeros':
                out = F.conv2d(x, self.vn_kernel, groups=self.channels, padding=d, dilation=d)
            else:
                x_pad = F.pad(x, (d,d,d,d), mode=self.padding_mode)
                out = F.conv2d(x_pad, self.vn_kernel, groups=self.channels, padding=0, dilation=d)
            perceptions.append(out)
        return torch.cat(perceptions, dim=1)

    # arg0 must be a list (even if input_length is 1)
    def forward(self, states: list[torch.Tensor], action_map: torch.Tensor|None, extra_map: torch.Tensor|None):
        # perceive each state and concat
        percepts = [self.perceive(s) for s in states]

        to_cat = [*percepts] 

        if self.use_global_context:
            global_ctx = states[0].mean(dim=[2,3], keepdim=True).expand_as(states[0])
            to_cat.append(global_ctx)
        
        if action_map is not None:
            to_cat.append(action_map)
        if extra_map is not None:
            to_cat.append(extra_map)

        # one single memory alloc
        inp = torch.cat(to_cat, dim=1)
        
        dx = self.net(inp)
        return states[0] + dx # apply delta to current state only
    
    # s + a = s'
    # e.g: (with input_length = 3): s + s' + s'' + a = s'''
    # where input_length >= 1
    def step(self, states: list[torch.Tensor], action_map: torch.Tensor|None, extra_map: torch.Tensor|None, microsteps: int):
        state_history = states  # [current, prev1, prev2, ...]
        
        for _ in range(microsteps):
            new_state = self(state_history, action_map, extra_map)
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
    def load_data(glob_files: str, limit: int|None=None, **kwargs) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        # load all data to RAM (more efficient than a DataLoader for small datasets)
        files = glob(glob_files)
        
        if limit is not None:
            files = files[:limit]
        
        # changed to be dynamic
        results = {name: [] for name in kwargs}
    
        for f in files:
            data = np.load(f)
            for name, dtype in kwargs.items():
                tensor = torch.from_numpy(data[name])

                if dtype == 'float':
                    tensor = tensor.float()
                elif dtype == 'long':
                    tensor = tensor.long()
                
                results[name].append(tensor)
        
        return tuple(results[name] for name in kwargs)

    # loads the state at file 'file', at index 'idx', and returns as ndarray, useful for interference
    @staticmethod
    def load_data_first(filename: str, idx: int=0) -> np.ndarray:
        states = np.load(filename)['states']
        return states[idx] # as tensor: torch.from_numpy(states[idx]).float()