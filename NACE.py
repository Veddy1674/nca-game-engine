import torch
import torch.nn as nn
import torch.nn.functional as F
from warnings import warn

from glob import glob
import numpy as np

class NACE(nn.Module):
    """
    A PyTorch implementation of a 'Neural Adaptive Cellular Engine', a model capable of learning emergent behaviors for game-like simulations.
    
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
        `custom_kernel`: Custom kernel as a list, to use for the perceive function (Note: this will override the default Von Neumann neighborhood and 'dilations' must be [1]; XY axis are flipped).
        `device`: Device to run the model on (default: cuda if available, else cpu).
    """
    def __init__(self, actions: int, vis_channels: int, hid_channels: int, extra_channels:int=0, hidden_neurons:int=128,
                 input_length:int=1, padding_mode: str = 'zeros', use_global_context: bool = False, dilations: list[int] = [1],
                 custom_kernel: list[list[int]]=None, device: str=None
                ):
        super().__init__()

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device
        self.padding_mode = padding_mode
        self.input_length = input_length
        self.use_global_context = use_global_context
        self.dilations = dilations

        self.actions = actions # can be 0 or >= 2, '1' makes no sense
        self.extra_channels = extra_channels # global, like actions

        self.vis_channels = vis_channels
        self.hid_channels = hid_channels # causes issues if '1', 0 or >= 2 is fine
        self.channels = vis_channels + hid_channels

        self.custom_kernel = custom_kernel

        if self.custom_kernel is not None:
            if self.dilations != [1]:
                warn("'dilations' has been set to [1] because a custom kernel was provided.", category=UserWarning)
                self.dilations = [1]

            self.kernel_size = sum(sum(row) for row in self.custom_kernel) # how many 1s in the kernel
        else:
            self.kernel_size = 5 # default with Von Neumann

        # cell sees what the kernel sees, + actions, global context, past inputs, extra channels and so on
        self.input_dim = ((self.channels * (self.kernel_size * len(self.dilations))) * self.input_length) + self.actions + self.extra_channels + (self.channels if self.use_global_context else 0)

        if self.hid_channels > 0 and (hidden_neurons / self.hid_channels) <= 2:
            print("Warning: 'hidden_neurons / hid_channels' is less than or equal to 2, this might cause instability.")

        # update net
        self.net = nn.Sequential(
            # hidden neurons / hidden channels > 2
            nn.Conv2d(self.input_dim, hidden_neurons, kernel_size=1),
            nn.LeakyReLU(0.01), # looks like it makes the net slightly more stable than ReLU
            nn.Conv2d(hidden_neurons, self.channels, kernel_size=1), # delta!
            nn.Hardtanh(-0.5, 0.5) # previously using SELU
        )

        # init to zero the second Conv2d for stability
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

        if self.custom_kernel is not None:
            # example with 5x5:
            # [
            #     [0, 0, 1, 0, 0],
            #     [0, 1, 1, 1, 0],
            #     [1, 1, 1, 1, 1],
            #     [0, 1, 1, 1, 0],
            #     [0, 0, 1, 0, 0]
            # ]
            # can be a rectangle and any size, as long as the two conditions below are met

            self.kernel_h = len(self.custom_kernel)
            self.kernel_w = len(self.custom_kernel[0])

            if self.kernel_h % 2 != 1 or self.kernel_w % 2 != 1:
                raise ValueError(f"Kernel must have odd dimensions, got {self.kernel_h}x{self.kernel_w}")
            
            if self.kernel_h < 3 or self.kernel_w < 3:
                raise ValueError(f"Kernel must be atleast 3x3, got {self.kernel_h}x{self.kernel_w}")

            kernels = torch.zeros(self.kernel_size, self.kernel_h, self.kernel_w)
            
            idx = 0
            for dy in range(self.kernel_h):
                for dx in range(self.kernel_w):
                    if self.custom_kernel[dy][dx] == 1:
                        kernels[idx, dy, dx] = 1.0
                        idx += 1
            
            if idx != self.kernel_size:
                raise ValueError("Custom kernel must only contain 0s and 1s.")
        
        else: # default Von Neumann neighborhood

            # these are not used in perceive(), but keeping for consistency
            self.kernel_h = 3
            self.kernel_w = 3

            kernels = torch.zeros(self.kernel_size, self.kernel_h, self.kernel_w)
            
            kernels[0, 1, 1] = 1.0 # center (self)
            kernels[1, 0, 1] = 1.0 # up
            kernels[2, 2, 1] = 1.0 # down
            kernels[3, 1, 0] = 1.0 # left
            kernels[4, 1, 2] = 1.0 # right

        # end

        self.register_buffer('kernel', kernels.unsqueeze(1).repeat(self.channels, 1, 1, 1))
        
        print(f"Using {self.device.upper()}")
        self.to(self.device)
    
    def perceive(self, x: torch.Tensor):
        # x shape is BCHW

        # for custom kernels that aren't 3x3
        if self.custom_kernel is not None and (self.kernel_h != 3 or self.kernel_w != 3):

            # pad and slide instead of F.conv2d is equivalent and way faster on huge kernels
            kh, kw = self.kernel_h, self.kernel_w
            cy, cx = kh // 2, kw // 2
            x_pad = F.pad(x, (cx, cx, cy, cy), mode='constant' if self.padding_mode == 'zeros' else self.padding_mode)

            slices = []
            for dy in range(kh):
                for dx in range(kw):
                    if self.custom_kernel[dy][dx]:
                        slices.append(x_pad[:, :, dy:dy+x.shape[2], dx:dx+x.shape[3]])
            
            return torch.cat(slices, dim=1)

        # for Von Neumann neighborhood or 3x3 custom kernels
        if len(self.dilations) == 1: # default, avoid concat's memory alloc
            d = self.dilations[0]
            # NOTE: If k_center WOULD BE defined for the default kernel, it would be (1, 1)
            # which means using k_center[0] or self.dilations[0] is the same thing, although using center would 'make more sense'

            if self.padding_mode == 'zeros':
                out = F.conv2d(x, self.kernel, groups=self.channels, padding=d, dilation=d)
            else:
                x_pad = F.pad(x, (d,d,d,d), mode=self.padding_mode)
                out = F.conv2d(x_pad, self.kernel, groups=self.channels, padding=0, dilation=d)

            return out

        perceptions = []

        for d in self.dilations:
            if self.padding_mode == 'zeros':
                out = F.conv2d(x, self.kernel, groups=self.channels, padding=d, dilation=d)
            else:
                x_pad = F.pad(x, (d,d,d,d), mode=self.padding_mode)
                out = F.conv2d(x_pad, self.kernel, groups=self.channels, padding=0, dilation=d)
            
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
    def load_data(glob_files: str, limit: int|tuple[int,int]|None=None, load_quick: bool=True, load_instant: bool=False, **kwargs) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        files = glob(glob_files)
        
        if limit is not None:
            if isinstance(limit, tuple):
                start, end = limit
                files = files[start:end] if end is not None else files[start:]
                
            elif isinstance(limit, int):
                files = files[:limit]
        
        if load_instant:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # load all data to RAM (more efficient than dataloaders for small datasets)
        if load_quick:
            
            results = {name: [] for name in kwargs}
        
            for f in files:
                data = np.load(f)
                for name, dtype in kwargs.items():
                    tensor = torch.from_numpy(data[name])

                    if dtype == 'float':
                        tensor = tensor.float()
                    elif dtype == 'long':
                        tensor = tensor.long()
                    
                    if load_instant:
                        tensor = tensor.to(device)
                    
                    results[name].append(tensor)
            
            return tuple(results[name] for name in kwargs)
        
        else: # lazy loading

            results = []
            
            for name, dtype in kwargs.items():
                results.append(Dataset(files, name, dtype))

            return tuple(results)

    # loads the state at file 'file', at index 'idx', and returns as ndarray, useful for inference
    @staticmethod
    def load_data_first(filename: str, idx: int=0) -> np.ndarray:
        states = np.load(filename)['states']
        return states[idx] # as tensor: torch.from_numpy(states[idx]).float()

class Dataset:
    def __init__(self, files, name, dtype):
        self.files = files
        self.name = name
        self.dtype = dtype
        self._lengths = None
        
    def __len__(self):
        if self._lengths is None:
            self._lengths = [len(np.load(f, mmap_mode='r')[self.name]) for f in self.files]
        
        return sum(self._lengths)
    
    def __getitem__(self, idx):
        # find file and idx
        for file_idx, f in enumerate(self.files):
            if self._lengths is None:
                len_f = len(np.load(f, mmap_mode='r')[self.name])
            else:
                len_f = self._lengths[file_idx]
                
            if idx < len_f:
                data = np.load(f, mmap_mode='r')[self.name][idx]

                if np.isscalar(data): # add axis if scalar
                    data = data[np.newaxis]

                tensor = torch.from_numpy(data)

                if self.dtype == 'float':
                    return tensor.float()
                elif self.dtype == 'long':
                    return tensor.long()
                else:
                    return tensor
                
            idx -= len_f
        raise IndexError