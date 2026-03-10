from NCA import NCA
import torch

# train params
STEPS = 400
BATCH_SIZE = 48 # increases total number of samples, along with VRAM usage
LOG_SEGMENTS = 10
# LOAD_MODEL = "mario/ccc.pt" # what train.py loads (comment out to train from scratch)
FILE_NAME = "mario/ccc.pt" # result model file name (post training)
DATA_GLOB = "mario/dev/ccc/ccc_*.npz"
MICROSTEPS = 8
TRAIN_STEPS = 1
POOL_LENGTH = None
EXTRA_MAPS = {}

# visualizer params
GRID_SIZE = (16, 16)
FIRST_DATA_FILE = "mario/dev/ccc/ccc_0.npz"
STARTING_IMAGE = None # repeated for each input_length for simplicity
BIT_DEPTH_LEVELS = 256

base_size = 896 # 16 x 56 (preferring multiples of 16 for pixel-perfect-looking graphics)

# auto resize so it looks pretty much always decent
# 'base_size' on the long side and proportional on the other (e.g: 16x8 -> 600,300 with base_size=600)
ratio = GRID_SIZE[0] / GRID_SIZE[1]
if ratio >= 1: # taller or equal
    WIN_SIZE = (int(base_size / ratio), base_size)
else: # larger
    WIN_SIZE = (base_size, int(base_size * ratio))

# or just force:
# WIN_SIZE = (600, 600)

# RGB
COLOR_MAP = { # RGB
    0: {'name': 'Black', 'color': [0, 0, 0]},
    1: {'name': 'Goomba1', 'color': [156, 74, 0]},
    2: {'name': 'Goomba2', 'color': [255, 206, 197]},
    3: {'name': 'Transparency', 'color': [255, 255, 255]}
}

# leave this so
if COLOR_MAP is not None:
    bgr_colormap = {k: v['color'][::-1] for k, v in COLOR_MAP.items()}

# q, y, r, esc are not allowed
KEY_MAP = {
    0: ord(' '),
}
DEFAULT_KEY = None
FPS = 30

model = NCA(actions=0, vis_channels=len(COLOR_MAP), hid_channels=0, hidden_neurons=64, padding_mode='zeros', device='cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

loss_func = torch.nn.CrossEntropyLoss()

# customizable function which defines the loss
def loss_calc(model_pred: torch.Tensor, actions: torch.Tensor, states: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pred_visible = model_pred[:, :model.vis_channels]
    return loss_func(pred_visible, targets.argmax(dim=1))