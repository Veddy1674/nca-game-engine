from NCA import NCA
import torch, torch.nn.functional as F
from types import MethodType

# train params
STEPS = 100
BATCH_SIZE = 128 # increases total number of samples, along with VRAM usage
LOG_SEGMENTS = 10
# LOAD_MODEL = "colorchange/colorchangeRGB.pt" # what train.py loads (comment out to train from scratch)
FILE_NAME = "colorchange/colorchangeRGB.pt" # result model file name (post training)
DATA_GLOB = "colorchange/dev/data/colorchangeRGB_*.npz"
MICROSTEPS = 1

# visualizer params
FIRST_DATA_FILE = "colorchange/dev/data/colorchangeRGB_0.npz"
GRID_SIZE = (32, 32) # H, W
STARTING_IMAGE = None # repeated for each input_length for simplicity
BIT_DEPTH_LEVELS = 4 # snap colors so that the AI doesn't predict slightly darker/brighter colors that get worse each step

base_size = 900

# auto resize so it looks pretty much always decent
# 'base_size' on the long side and proportional on the other (e.g: 16x8 -> 600,300 with base_size=600)
ratio = GRID_SIZE[0] / GRID_SIZE[1]
if ratio >= 1: # taller or equal
    WIN_SIZE = (int(base_size / ratio), base_size)
else: # larger
    WIN_SIZE = (base_size, int(base_size * ratio))

# or just force:
# WIN_SIZE = (600, 600)

# each local object is a class/channel, if two different objects have the same color, that's still two classes
COLOR_MAP = None # for RGBA it's unused

# q, y, r, esc are not allowed
KEY_MAP = {
    1: ord(' ')
}
DEFAULT_KEY = None
FPS = 1

model = NCA(actions=0, vis_channels=3, hid_channels=0, input_length=1, device='cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)

loss_func = torch.nn.MSELoss()

# change the way the model perceives borders, instead of seeing 0.0, it sees itself
def perceive_replicate(self, x):
    x_padded = F.pad(x, (1, 1, 1, 1), mode='replicate') #!
    return F.conv2d(x_padded, self.vn_kernel, groups=self.channels, padding=0)

model.perceive = MethodType(perceive_replicate, model)

# customizable function which defines the loss
def loss_calc(pred_visible: torch.Tensor, states: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return loss_func(pred_visible, targets) # for RGB