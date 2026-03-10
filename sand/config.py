from NCA import NCA
import torch

# train params
STEPS = 1_000
BATCH_SIZE = 12 # increases total number of samples, along with VRAM usage
LOG_SEGMENTS = 10
# LOAD_MODEL = "sand/sand.pt" # what train.py loads (comment out to train from scratch)
FILE_NAME = "sand/sand2.pt" # result model file name (post training)
DATA_GLOB = "sand/dev/data/sand_*.npz"
MICROSTEPS = 12

# visualizer params
FIRST_DATA_FILE = "sand/dev/data/sand_0_26116.npz"
GRID_SIZE = (128, 128) # H, W
STARTING_IMAGE = "sand/start_img2.png" # repeated for each input_length for simplicity
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

model = NCA(actions=4, vis_channels=len(COLOR_MAP), hid_channels=16, input_length=2, padding_mode='zeros', device='cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)

# the way it worked the best for me was, training on weights focused on floor, rect, background and sand weighted 0
# and then fine-tuned to 0.1, 0.5, 6.0, 4.0 or similar, to focus on the sand physics (while making sure it doesn't forget rect and floor)

# floor, rect, sand, background
weights = torch.tensor([0.1, 1.0, 6.0, 4.0], device=model.device)
loss_func = torch.nn.CrossEntropyLoss(weight=weights)

# customizable function which defines the loss
def loss_calc(model_pred: torch.Tensor, actions: torch.Tensor, states: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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
        h, w = img.shape[:2]
        
        # gradient in BGR
        color_start = np.array([71, 169, 191])
        color_end = np.array([38, 38, 255])
        
        # gradient on column
        y_indices, x_indices = np.where(sand_mask)
        t = x_indices / (w - 1) # normalize
        
        # lerp colors
        colors = (color_start * (1 - t[:, np.newaxis]) + color_end * t[:, np.newaxis]).astype(np.uint8)
        
        result[y_indices, x_indices] = colors
    
    return result