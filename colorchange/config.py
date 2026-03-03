from NCA import NCA
import torch

# train params
STEPS = 100
BATCH_SIZE = 128 # increases total number of samples, along with VRAM usage
LOG_SEGMENTS = 10
# LOAD_MODEL = "colorchange/colorchange.pt" # what train.py loads (comment out to train from scratch)
FILE_NAME = "colorchange/colorchange.pt" # result model file name (post training)
DATA_GLOB = "colorchange/dev/data/colorchange_*.npz"
MICROSTEPS = 1

# visualizer params
FIRST_DATA_FILE = "colorchange/dev/data/colorchange_0.npz"
GRID_SIZE = (32, 32) # H, W
STARTING_IMAGE = None # repeated for each input_length for simplicity
BIT_DEPTH_LEVELS = 256

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
COLOR_MAP = { # RGB
    0: {'name': 'White', 'color': [255, 255, 255]},
    1: {'name': 'Green', 'color': [0, 255, 0]},
    2: {'name': 'Red',   'color': [255, 0, 0]},
    # in this case, even though blue is the 'death' channel (because it's the last),
    # it works just like the other channels, and won't confuse the model
    3: {'name': 'Blue',  'color': [0, 0, 255]}
}

# leave this so
if COLOR_MAP is not None:
    bgr_colormap = {k: v['color'][::-1] for k, v in COLOR_MAP.items()}

# q, y, r, esc are not allowed
KEY_MAP = {
    1: ord(' ')
}
DEFAULT_KEY = None
FPS = 1

model = NCA(actions=0, vis_channels=len(COLOR_MAP), hid_channels=0, input_length=1, device='cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)

loss_func = torch.nn.CrossEntropyLoss()

# customizable function which defines the loss
def loss_calc(pred_visible: torch.Tensor, states: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # loss ONLY on visible channels
    return loss_func(pred_visible, targets.argmax(dim=1)) # for one-hot