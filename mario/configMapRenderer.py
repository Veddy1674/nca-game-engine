from NCA import NCA
import torch, torch.nn.functional as F

# train params
STEPS = 3000
BATCH_SIZE = 64 # increases total number of samples, along with VRAM usage
LOG_SEGMENTS = 100
# LOAD_MODEL = "mario/renderer_small.pt" # what train.py loads (comment out to train from scratch)
FILE_NAME = "mario/renderer_small.pt" # result model file name (post training)
DATA_GLOB = "mario/dev/data/mario_*.npz"
MICROSTEPS = 12
TRAIN_STEPS = 1
POOL_LENGTH = None
EXTRA_MAPS = {}
LOSS_GRAPH = "mario/loss_graph.png" # can be None

# visualizer params
GRID_SIZE = (15, 15)
FIRST_DATA_FILE = "mario/dev/data/mario_0.npz"
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

from mario.dev.env import GAME_SPRITES, ANIM_GAME_SPRITES, COLOR_MAP_LIST
from mario.dev.make_sprites import SPRITE_SIZE

# RGB
COLOR_MAP = {i: {'name': f'Color{i}', 'color': color}
            for i, color in enumerate(COLOR_MAP_LIST)}

"""Result will be something like this:
COLOR_MAP = {
    0: {'name': 'Color0', 'color': [r, g, b]},
    1: {'name': 'Color1', 'color': [r, g, b]},
    ...
}"""

# init all to 1.0
weights = [1.0 for _ in range(len(COLOR_MAP))]

# normally the weights would be much lower, but everything else is 1.0
weights[0] = 0.8 # sky
weights[35] = 0.2 # ground

# to see which index corresponds to which sprite: (in env.py)
# Image.fromarray(GAME_SPRITES[idx]).show()
# exit()

# leave this so
if COLOR_MAP is not None:
    bgr_colormap = {k: v['color'][::-1] for k, v in COLOR_MAP.items()}

# q, y, r, esc are not allowed
KEY_MAP = {
    1: ord('d'), # right
    2: ord('a'), # left
}
DEFAULT_KEY = 0
FPS = None # waits for input
FPS_PYGAME = 60
REFRESH_RATE = 100
HIDE_INFO = True # to record

scroll_speed = 16

model = NCA(actions=3, vis_channels=len(COLOR_MAP), use_global_context=True, hid_channels=16, hidden_neurons=256, padding_mode='zeros', device='cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # (1e-4 if fine-tuning, to avoid catastrophic forgetting)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=6e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.9)

loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, device=model.device))

# customizable function which defines the loss
def loss_calc(model_pred: torch.Tensor, actions: torch.Tensor, states: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pred_visible = model_pred[:, :model.vis_channels]
    target_classes = targets.argmax(dim=1)
    
    main_loss = loss_func(pred_visible, target_classes)
    
    # extra penalty on rightmost and leftmost column to incentivate level generation
    last_col_loss = F.cross_entropy(
        pred_visible[:, :, :, -1].permute(0, 2, 1).reshape(-1, model.vis_channels),
        target_classes[:, :, -1].reshape(-1),
    )

    first_col_loss = F.cross_entropy(
        pred_visible[:, :, :, 0].permute(0, 2, 1).reshape(-1, model.vis_channels),
        target_classes[:, :, 0].reshape(-1),
    )
    
    return main_loss + 0.7 * last_col_loss + 0.7 * first_col_loss

import numpy as np, cv2

global_timer = 0
qmark_cyle = [0, 1, 2, 1, 0] # animation like the original game
qmark_state = 0 # idx of cycle
increment_every = int(8 * (REFRESH_RATE / 50)) # 8 if refresh rate is 50 (simply tested what looks good)

firsttime = True
def _render(state: np.ndarray) -> np.ndarray:
    global firsttime, model, qmark_cyle

    if firsttime:
        firsttime = False
        # compile model ONLY during interference
        print("Compiling model for faster interference...")
        model = torch.compile(model, mode="reduce-overhead", backend="cudagraphs")
        print("Done!")

    color = state.argmax(axis=0) # (H, W)

    # corresponding sprite for that color
    img: np.ndarray = GAME_SPRITES[color] # (H, W, SPRITE_SIZE, SPRITE_SIZE, 3)

    # set question mark animation
    mask = (color == 11)
    if np.any(mask):
        img[mask] = ANIM_GAME_SPRITES[qmark_cyle[qmark_state]]

    h, w = color.shape # 15x15
    img = img.transpose(0, 2, 1, 3, 4).reshape(h * SPRITE_SIZE, w * SPRITE_SIZE, 3) # (H*16, W*16, 3), 16 is sprite size (15 * 16 = 240x240)

    return img[:, :, ::-1] # 240x240 RGB

def _render_smooth():
    global stateA, stateB, scroll_t

    if np.array_equal(stateA, stateB): # for example if trying to go left at the beginning of the level
        scroll_t = 0 # important
        stateB = None
        return _render(stateA)

    # render both states
    imgA = _render(stateA)
    imgB = _render(stateB)

    # create new image
    img = np.zeros_like(imgA)
    W = imgA.shape[1]

    at = abs(scroll_t)
    offset = SPRITE_SIZE - at # offset of state B

    if scroll_t > 0: # right - stateA to left, stateB to right
        img[:, :W - at] = imgA[:, at:]
        img[:, offset:] = imgB[:, :W - offset]
    else: # left - stateA to right, stateB to left
        img[:, at:] = imgA[:, :W - at]
        img[:, :W - offset] = imgB[:, offset:]

    return img

stateA = None
stateB = None
scroll_t = 0
pending_action = None

def state_to_img(state: np.ndarray) -> np.ndarray:
    global stateA, global_timer, qmark_state, qmark_cyle

    global_timer += 1
    if global_timer % increment_every == 0: # usually update every 8 frames (like original game logic)
        qmark_state = (qmark_state + 1) % len(qmark_cyle)
        # print(qmark_state)

    if stateA is None: # first frame
        stateA = state.copy()

    img = _render_smooth() if stateB is not None else _render(stateA)

    return cv2.resize(img, WIN_SIZE, interpolation=cv2.INTER_NEAREST)

def manage_actions(action, state_history, snap_colors, predict_next):
    global stateA, stateB, scroll_t, pending_action

    if action == 1:
        scroll_t += scroll_speed
    elif action == 2:
        scroll_t -= scroll_speed

    # returned to origin - clear stateB
    if scroll_t == 0 and stateB is not None:
        stateB = None
        pending_action = None

    # generate next state once per action
    if stateB is None and action != 0:
        _, next_frame = predict_next(state_history, action)
        # print("NCA Forward")

        next_frame = np.eye(model.vis_channels)[next_frame].transpose(2, 0, 1)
        
        state_history.append(next_frame)
        if len(state_history) > model.input_length:
            state_history.pop(0)

        stateB = next_frame
        pending_action = action

    # snap to stateB
    if scroll_t >= SPRITE_SIZE or scroll_t <= -SPRITE_SIZE:
        stateA = stateB.copy()
        stateB = None

        pending_action = None
        scroll_t = 0

    # clear stateB
    if scroll_t == 0 and stateB is not None:
        stateB = None
        pending_action = None

    return None, stateA