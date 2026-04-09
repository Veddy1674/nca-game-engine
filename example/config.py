if __name__ == "__main__":
    print("This file is not meant to be run directly.")
    exit()

from NACE import NACE
import torch

# details about parameters can be found in configs_vars.py

# train params
STEPS = 1_000
BATCH_SIZE = 256
LOG_SEGMENTS = 100

# LOAD_MODEL = "example/example.pt"
FILE_NAME = "example/example.pt"
DATA_GLOB = "example/dev/data/example_*.npz"
MICROSTEPS = 2

TRAIN_STEPS = 1
# WEIGHT_LOSS = 1 # by default: linear if TRAIN_STEPS <= 3, squared if TRAIN_STEPS <= 7, else exponential
# GRADIENT_CLIP = 1.0 # keep to 1.0 unless TRAIN_STEPS and WEIGHT_LOSS are high, auto-defined by default but not 100% safe
POOL_LENGTH = None
LOAD_QUICK = True
LOAD_INSTANT = True
FILES_INCLUDE = None # include every file of the dataset (only one in this case)
EXTRA_MAPS = {}
LOSS_GRAPH = "example/loss_graph.png" # can be None
LOAD_OPTIMIZER = True # false = override with model lr

# visualizer params
MODEL_PATH = "example/example.pt"
FIRST_DATA_FILE = "example/dev/data/example_0.npz"
GRID_SIZE = (16, 16) # H, W (testing in a bigger grid than the training one to show generalization)
STARTING_IMAGE = None
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

COLOR_MAP = { # RGB
    0: {'name': 'Player',     'color': [240, 240, 240]},
    1: {'name': 'Background', 'color': [33, 33, 33]}
}

# player, background
weights = [1.0, 0.02]
# in a 8x8 grid, player is 1/64 of the image, so here background is
# weighted a little over 1/64 of player, to push the model to learn the player movement quicker

if COLOR_MAP is not None:
    bgr_colormap = {k: v['color'][::-1] for k, v in COLOR_MAP.items()}

# q, y, r, esc are reserved keys, do not use them here
KEY_MAP = {
    0: ord('w'),
    1: ord('s'),
    2: ord('a'),
    3: ord('d'),
}
DEFAULT_KEY = None
FPS = None # waits for input

TEMPERATURE = 1.0
TOP_P = 0.99 # only applied if temperature > 1.0

model = NACE(actions=4, vis_channels=len(COLOR_MAP), hid_channels=0, input_length=1, hidden_neurons=24, padding_mode='circular', device='cuda')
print("Input dimension:", model.input_dim) # you could concatenate: "->", model.projection_channels, when using projection

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# simple scheduler for a simple environment
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)

weights = torch.tensor(weights, device=model.device)
loss_func = torch.nn.CrossEntropyLoss(weight=weights)

# customizable function which defines the loss
def loss_calc(model_pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # loss ONLY on visible channels
    pred_visible = model_pred[:, :model.vis_channels]
    return loss_func(pred_visible, targets.argmax(dim=1)) # for one-hot