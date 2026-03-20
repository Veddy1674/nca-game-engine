from NCA import NCA
import torch

# train params
STEPS = 1_500
BATCH_SIZE = 12 # increases total number of samples, along with VRAM usage
LOG_SEGMENTS = 100
# LOAD_MODEL = "sand/sand.pt" # what train.py loads (comment out to train from scratch)
FILE_NAME = "sand/sand.pt" # result model file name (post training)
DATA_GLOB = "sand/dev/data/sand_*.npz"
MICROSTEPS = 4
TRAIN_STEPS = 1
POOL_LENGTH = None
LOAD_QUICK = True # wheter to load all the datasets to RAM or use lazy loading
LOAD_INSTANT = True # wheter to load everything to VRAM right away (for small datasets where CPU is the bottleneck)
EXTRA_MAPS = {}
LOSS_GRAPH = "sand/loss_graph.png" # can be None

# visualizer params
FIRST_DATA_FILE = "sand/dev/data/sand_0_43962.npz"
GRID_SIZE = (96, 96) # H, W
STARTING_IMAGE = "sand/start_img.png" # repeated for each input_length for simplicity
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
    0: {'name': 'Floor',      'color': [22, 22, 22]},
    1: {'name': 'Rect',       'color': [119, 119, 119]}, # sand spawner
    2: {'name': 'Sand',       'color': [193, 171, 73]},
    3: {'name': 'Background', 'color': [38, 38, 38]}
}

# leave this so
if COLOR_MAP is not None:
    bgr_colormap = {k: v['color'][::-1] for k, v in COLOR_MAP.items()}

# q, y, r, esc are not allowed
KEY_MAP = {
    1: ord('d'), # right
    2: ord('a'), # left
    3: ord(' ') # spawn
}
DEFAULT_KEY = 0 # noop
FPS = 60

TEMPERATURE = 1.0
TOP_P = 0.99

LOAD_OPTIMIZER = True

# surprisingly, hidden channels=0, hidden_neurons=64 and input_lenght=1 work as well
# as a overparameterized model with 16 hidden channels, 128 hidden neurons and 2 input length.
model = NCA(actions=4, vis_channels=len(COLOR_MAP), hid_channels=0, input_length=1, hidden_neurons=64, padding_mode='zeros', device='cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999) # with 1500 steps, from 1e-3 to 8.6e-4

# fine-tuning with sand weight to zero and then back up works fine, but I found this combination to work as well and without fine-tuning
# floor, rect, sand, background
weights = torch.tensor([0.1, 0.5, 1.7, 1.5], device=model.device)
loss_func = torch.nn.CrossEntropyLoss(weight=weights)

# customizable function which defines the loss
def loss_calc(model_pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pred_visible = model_pred[:, :model.vis_channels] # loss on visible channels
    return loss_func(pred_visible, targets.argmax(dim=1))

import numpy as np

# img is the state after one-hot to BGR conversion, but before cv2 upscaling
def post_processing(img: np.ndarray) -> np.ndarray:
    result = img.copy()
    
    sand_color = np.array(bgr_colormap[2])
    
    # find sand
    sand_mask = np.all(result == sand_color, axis=2)
    
    if np.any(sand_mask):
        w = img.shape[1] # h, w

        # gradient in BGR
        color_start = np.array([71, 169, 191]) # dark yellow
        color_end = np.array([38, 38, 255]) # bright red
        
        # gradient on column
        y_indices, x_indices = np.where(sand_mask)
        t = x_indices / (w - 1) # normalize
        
        # lerp colors
        colors = (color_start * (1 - t[:, np.newaxis]) + color_end * t[:, np.newaxis]).astype(np.uint8)
        
        result[y_indices, x_indices] = colors
    
    return result