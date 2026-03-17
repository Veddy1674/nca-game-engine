from NCA import NCA
import torch

# train params
STEPS = 400
BATCH_SIZE = 512 # increases total number of samples, along with VRAM usage
LOG_SEGMENTS = 100
# LOAD_MODEL = "snake/snake_new.pt" # what train.py loads (comment out to train from scratch)
FILE_NAME = "snake/snake_new.pt" # result model file name (post training)
DATA_GLOB = "snake/dev/data/snake_*.npz"
MICROSTEPS = 4
TRAIN_STEPS = 1
POOL_LENGTH = None
LOAD_QUICK = True # wheter to load all the datasets to RAM or use lazy loading
LOAD_INSTANT = True # wheter to load everything to VRAM right away (for small datasets where CPU is the bottleneck)
EXTRA_MAPS = {}
LOSS_GRAPH = "snake/snake_new.png" # can be None

# visualizer params
FIRST_DATA_FILE = "snake/dev/data/snake_0.npz"
GRID_SIZE = (16, 16) # H, W
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

weights = [0.9, 1.2, 0, 0.9]

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
FPS = None # waits for input
FPS_PYGAME = 60
REFRESH_RATE = 100
HIDE_INFO = False
VSYNC = True

TEMPERATURE = 1.0
TOP_P = 0.99 # only applied if temperature > 1.0

LOAD_OPTIMIZER = False # if False uses the lr defined below, otherwise continues from the loaded model's lr

model = NCA(actions=4, vis_channels=len(COLOR_MAP), dilations=[1, 2], hid_channels=16, input_length=1, hidden_neurons=192, padding_mode='zeros', device='cuda')
print(model.input_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)

# body, head, apple, background
weights = torch.tensor(weights, device=model.device)
loss_func = torch.nn.CrossEntropyLoss(weight=weights)

# customizable function which defines the loss
def loss_calc(model_pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # loss ONLY on visible channels
    pred_visible = model_pred[:, :model.vis_channels]
    return loss_func(pred_visible, targets.argmax(dim=1)) # for one-hot

# def addnoise(current_x):
#     if torch.rand(1).item() < 0.3:
#         noise = torch.randn_like(current_x[0][:, :model.vis_channels]) * 0.05
#         current_x[0][:, :model.vis_channels] += noise