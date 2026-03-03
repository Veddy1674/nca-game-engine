from NCA import NCA
import torch

# train params
STEPS = 1_000
BATCH_SIZE = 128 # increases total number of samples, along with VRAM usage
LOG_SEGMENTS = 10
# LOAD_MODEL = "snake/snake_decent.pt" # what train.py loads (comment out to train from scratch)
FILE_NAME = "snake/snake_decent.pt" # result model file name (post training)
DATA_GLOB = "snake/dev/data/snake_*.npz"
MICROSTEPS = 6

# visualizer params
FIRST_DATA_FILE = "snake/dev/data/snake_0.npz"
GRID_SIZE = (8, 8) # H, W
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
    0: {'name': 'Snake Body', 'color': [0, 255, 0]},
    1: {'name': 'Snake Head', 'color': [0, 154, 0]},
    2: {'name': 'Apple',      'color': [255, 0, 0]},
    3: {'name': 'Background', 'color': [30, 30, 30]}
}

# leave this so
if COLOR_MAP is not None:
    bgr_colormap = {k: v['color'][::-1] for k, v in COLOR_MAP.items()}

# q, y, r, esc are not allowed
KEY_MAP = {
    0: ord('w'),
    1: ord('s'),
    2: ord('a'),
    3: ord('d'),
}
DEFAULT_KEY = None
FPS = 1

model = NCA(actions=4, vis_channels=len(COLOR_MAP), hid_channels=12, input_length=1, device='cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)

# body, head, apple, background
weights = torch.tensor([2.5, 10.0, 4.0, 5.5], device=model.device)
loss_func = torch.nn.CrossEntropyLoss(weight=weights)

# customizable function which defines the loss
def loss_calc(pred_visible: torch.Tensor, states: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # loss ONLY on visible channels
    return loss_func(pred_visible, targets.argmax(dim=1)) # for one-hot