from NACE import NACE
import torch, torch.nn.functional as F
from PIL import Image as newImage

# train params
STEPS = 15_000
BATCH_SIZE = 64 # increases total number of samples, along with VRAM usage
LOG_SEGMENTS = 100
# LOAD_MODEL = "mario/renderer3.pt" # what train.py loads (comment out to train from scratch)
FILE_NAME = "mario/renderer.pt" # result model file name (post training)
DATA_GLOB = "mario/dev/data/mario_*.npz"
MICROSTEPS = 15
TRAIN_STEPS = 1
POOL_LENGTH = None
LOAD_QUICK = True # wheter to load all the datasets to RAM or use lazy loading
LOAD_INSTANT = False # wheter to load everything to VRAM right away (for small datasets where CPU is the bottleneck)
EXTRA_MAPS = {'map_id': 'long'}
LOSS_GRAPH = "mario/loss_graph.png" # can be None

# visualizer params
GRID_SIZE = (15, 15)
FIRST_DATA_FILE = "mario/dev/data/mario_0.npz"
STARTING_IMAGE = "mario/dev/start_img.png" # repeated for each input_length for simplicity
BIT_DEPTH_LEVELS = 256

base_size = 960 # 240 x 4 (preferring multiples of 16x15 for pixel-perfect-looking graphics)

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

weights[-1] = 0.8 # sky
weights[35] = 0.4 # ground

# to see which index corresponds to which sprite: (in env.py)
# Image.fromarray(GAME_SPRITES[idx]).show()
# exit()

# leave this so
if COLOR_MAP is not None:
    bgr_colormap = {k: v['color'][::-1] for k, v in COLOR_MAP.items()}

# q, y, r, esc are not allowed
KEY_MAP = {
    1: ord('d'), # right
    # 2: ord('a'), # left
}
DEFAULT_KEY = None
FPS = None # waits for input
FPS_PYGAME = 60
REFRESH_RATE = 100
HIDE_INFO = True # to record
VSYNC = True

TEMPERATURE = 1.01
level = 4
TOP_P = 0.99 # only applied if temperature > 1.0

scroll_speed = 16 # pixels/frame, max 16

LOAD_OPTIMIZER = True # if False uses the lr defined below, otherwise continues from the loaded model's lr

model: NACE = NACE(actions=0, vis_channels=len(COLOR_MAP), use_global_context=True, hid_channels=0, extra_channels=13, hidden_neurons=128, padding_mode='zeros', device='cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999) # 0.9995

loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, device=model.device))

# customizable function which defines the loss
def loss_calc(model_pred: torch.Tensor, actions: torch.Tensor, states: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pred_visible = model_pred[:, :model.vis_channels]
    target_classes = targets.argmax(dim=1)
    
    main_loss = loss_func(pred_visible, target_classes)
    
    # extra penalty on rightmost and leftmost column to incentivate level generation
    last_col_loss = F.cross_entropy( # rightmost
        pred_visible[:, :, :, -1].permute(0, 2, 1).reshape(-1, model.vis_channels),
        target_classes[:, :, -1].reshape(-1),
    )

    # first_col_loss = F.cross_entropy( # leftmost
    #     pred_visible[:, :, :, 0].permute(0, 2, 1).reshape(-1, model.vis_channels),
    #     target_classes[:, :, 0].reshape(-1),
    # )
    
    return main_loss + 0.7 * last_col_loss# + 0.7 * first_col_loss

def addnoise(current_x):
    if torch.rand(1).item() < 0.3:
        noise = torch.randn_like(current_x[0][:, :model.vis_channels]) * 0.05
        current_x[0][:, :model.vis_channels] += noise

import numpy as np, cv2

global_timer = 0
# qmark anim indexes are: 12, 13, 14
qmark_cyle = [12, 13, 14, 13, 12] # animation like the original game
qmark_state = 0 # idx of cycle
qmark_increment_every = int(8 * (REFRESH_RATE / 50)) # 8 if refresh rate is 50 (simply tested what looks good)

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

def resetAll():
    global stateA, stateB, scroll_t, pending_action
    stateA = None
    stateB = None
    scroll_t = 0
    pending_action = None

resetAll()

def state_to_img(state: np.ndarray) -> np.ndarray:
    global stateA, global_timer, qmark_state, qmark_cyle

    global_timer += 1
    if global_timer % qmark_increment_every == 0: # usually update every 8 frames (like original game logic)
        qmark_state = (qmark_state + 1) % len(qmark_cyle)
        # print(qmark_state)

    if stateA is None: # first frame
        stateA = state.copy()

    # (240, 240, 3)
    img = _render_smooth() if stateB is not None else _render(stateA)

    return cv2.resize(img, WIN_SIZE, interpolation=cv2.INTER_NEAREST)

def manage_actions(action, state_history, snap_colors, predict_next, apply_top_p):
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
        _, next_frame = predict_next(state_history, action, apply_top_p)
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

# implemented resetAll()
def reset(maybe_resize, states_data, data_grid):
    if STARTING_IMAGE is not None:
        # load image RGB
        img = newImage.open(STARTING_IMAGE).convert("RGB")
        img = np.array(img)

        # resize if needed
        if img.shape[:2] != GRID_SIZE:
            img = cv2.resize(img, (GRID_SIZE[1], GRID_SIZE[0]), interpolation=cv2.INTER_NEAREST)

        h, w = img.shape[:2]
        state = np.zeros((model.vis_channels, h, w), dtype=np.float32)

        # default background (last channel)
        state[model.vis_channels - 1] = 1.0

        for cls_idx, data in COLOR_MAP.items():
            if cls_idx == (model.vis_channels - 1):
                continue

            color = np.array(data['color'], dtype=np.uint8)
            mask = np.all(img == color, axis=2)
            state[:, mask] = 0.0
            state[cls_idx, mask] = 1.0

        # init state_history with starting image repeated for simplicity
        state_history = [state.copy() for _ in range(model.input_length)]

        print(f"Loaded starting image: {STARTING_IMAGE}")

    else:
        state, success = maybe_resize(states_data[model.input_length - 1])
        if success:
            print(f"Mismatch found: Trained on {data_grid[0]}x{data_grid[1]}, visualizing on {GRID_SIZE[0]}x{GRID_SIZE[1]} {WIN_SIZE}")

        # oldest -> newest, so [-1] is always most recent (consistent with append)
        state_history = [maybe_resize(states_data[i])[0] for i in range(model.input_length)]

    resetAll()

    return state, state_history

# extra channel management
def predict_next(state_history: list[np.ndarray], action: int, apply_top_p):
    hidden = torch.zeros(1, model.hid_channels, *GRID_SIZE, device=model.device)

    # build model_x list from history
    model_x = []

    for k in range(model.input_length):
        s = state_history[-(k+1)] if k < len(state_history) else state_history[0] # pad with oldest
        s = torch.from_numpy(s).float().unsqueeze(0).to(model.device)
        s = torch.cat([s, hidden], dim=1)

        model_x.append(s)

    if model.actions > 1:
        action_map = torch.zeros(1, model.actions, *GRID_SIZE, device=model.device)
        action_map[0, action] = 1.0
    else:
        action_map = None
    
    extra_map = torch.zeros(1, model.extra_channels, *GRID_SIZE, device=model.device)
    extra_map[0, level, :, :] = 1.0

    with torch.no_grad(): # ignoring extra map, override predict_next in a config to implement
        pred = model.step(model_x, action_map, extra_map, microsteps=MICROSTEPS)

    logits = pred[0, :model.vis_channels]

    if COLOR_MAP is not None: # one-hot
        if TEMPERATURE <= 1.0: # use argmax
            return pred, logits.argmax(dim=0).cpu().numpy() # next_frame

        C, H, W = logits.shape
        logits = (logits / TEMPERATURE).view(C, -1).t() # apply temperature
        
        logits = apply_top_p(logits, p=TOP_P)
        probs = F.softmax(logits, dim=-1) # probabilities
        
        next_frame = torch.multinomial(probs, 1).view(H, W).cpu().numpy()
    else:
        next_frame = logits.cpu().numpy() # RGBA
    
    return pred, next_frame