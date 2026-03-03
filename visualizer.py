import cv2
import numpy as np
import torch
from PIL import Image as newImage

from sand.config import *

model.load("sand/sand.pt")
model.eval()

# only used for models that output RGBA!
def snap_colors(img: np.ndarray) -> np.ndarray:
    # img: float 0-1
    img = np.clip(img.astype(np.float32), 0.0, 1.0)
    levels = float(BIT_DEPTH_LEVELS)
    return np.round(img * (levels - 1)) / (levels - 1)

if 'state_to_img' not in globals():
    def state_to_img(state: np.ndarray):
        if COLOR_MAP is not None: # one-hot

            # create empty BGR image (all colors are black)
            img = np.zeros((*GRID_SIZE, 3), dtype=np.uint8)
            state = state.argmax(axis=0) # convert one-hot to class indices (0, 1, 2, 3)

            for cls, color in bgr_colormap.items():
                mask = (state == cls)
                img[mask] = color

            img = post_processing(img)
            
            return cv2.resize(img, WIN_SIZE, interpolation=cv2.INTER_NEAREST)
        
        else: # RGBA
            img = np.transpose(state[:3], (1, 2, 0)) # RGB
            img = np.clip(img * 255, 0, 255).astype(np.uint8) # clipping is necessary to avoid RuntimeWarning

            img = post_processing(img)

            return cv2.resize(img, WIN_SIZE, interpolation=cv2.INTER_NEAREST)

# for input_length >= 1:
if 'predict_next' not in globals():
    def predict_next(state_history: list[np.ndarray], action: int):
        hidden = torch.zeros(1, model.hid_channels, *GRID_SIZE, device=model.device)

        # build model_x list from history
        model_x = []

        for k in range(model.input_length):
            s = state_history[-(k+1)] if k < len(state_history) else state_history[0]  # pad with oldest
            s = torch.from_numpy(s).float().unsqueeze(0).to(model.device)
            s = torch.cat([s, hidden], dim=1)

            model_x.append(s)

        if model.actions > 1:
            action_map = torch.zeros(1, model.actions, *GRID_SIZE, device=model.device)
            action_map[0, action] = 1.0
        else:
            action_map = None

        with torch.no_grad():
            pred = model.step(model_x, action_map, microsteps=MICROSTEPS)

        if COLOR_MAP is not None: #! 4 or model.vis_channels?
            next_frame = pred[0, :4].argmax(dim=0).cpu().numpy() # one-hot
        else:
            next_frame = pred[0, :4].cpu().numpy() # RGBA
        
        return pred, next_frame

win_name = "NCA Visualizer"

states_data = np.load(FIRST_DATA_FILE)['states'] # e.g: (2002, 4, 8, 8), so it's steps, color channels, height, width
data_grid = states_data.shape[2], states_data.shape[3] # (H, W)

def maybe_resize(s):
    # if GRID_SIZE is different than trained data, resize (to test model on different grid sizes)
    if data_grid != GRID_SIZE:
        # expand background rather than resize
        c, h, w = s.shape

        new = np.zeros((c, GRID_SIZE[0], GRID_SIZE[1]), dtype=s.dtype)
        new[model.vis_channels - 1, :, :] = 1.0 # fill with background (last channel)
        new[:, :h, :w] = s # paste original in top-left!

        return new, True # true if has been resized
    
    return s, False

if 'manage_actions' not in globals():
    def manage_actions(action, state_history, snap_colors):
        last_prediction, next_frame = predict_next(state_history, action)

        if COLOR_MAP is not None:
            next_frame = np.eye(4)[next_frame].transpose(2, 0, 1)  # to one-hot (4,8,8)
        else:
            # color clip
            if BIT_DEPTH_LEVELS != 256:
                # avoid snap colors on alpha if present
                next_frame[:3] = snap_colors(next_frame[:3])  # snap_colors handles clip internally

        state_history.append(next_frame)
        
        if len(state_history) > model.input_length:
            state_history.pop(0)
        
        return last_prediction, next_frame

if 'post_processing' not in globals():
    def post_processing(state: np.ndarray) -> np.ndarray:
        return state

def reset():
    global state, frame_counter, last_prediction, state_history
    
    if STARTING_IMAGE is not None:
        # load image RGB
        img = newImage.open(STARTING_IMAGE).convert("RGB")
        img = np.array(img)

        # resize if needed
        if img.shape[:2] != GRID_SIZE:
            img = cv2.resize(img, (GRID_SIZE[1], GRID_SIZE[0]), interpolation=cv2.INTER_NEAREST)

        h, w = img.shape[:2]
        state = np.zeros((4, h, w), dtype=np.float32)

        if COLOR_MAP is not None:

            # default background (last channel)
            state[model.vis_channels - 1] = 1.0

            for cls_idx, data in COLOR_MAP.items():
                if cls_idx == (model.vis_channels - 1):
                    continue

                color = np.array(data['color'], dtype=np.uint8)
                mask = np.all(img == color, axis=2)
                state[:, mask] = 0.0
                state[cls_idx, mask] = 1.0
        
        else: # RGBA

            state[:3] = img.astype(np.float32) / 255 # RGB channels
            state[3] = 1.0 # alpha

        # init state_history with starting image repeated for simplicity
        state_history = [state.copy() for _ in range(model.input_length)]

        print(f"Loaded starting image: {STARTING_IMAGE}")


    else:
        state, success = maybe_resize(states_data[model.input_length - 1])
        if success:
            print(f"Mismatch found: Trained on {data_grid[0]}x{data_grid[1]}, visualizing on {GRID_SIZE[0]}x{GRID_SIZE[1]} {WIN_SIZE}")

        # oldest -> newest, so [-1] is always most recent (consistent with append)
        state_history = [maybe_resize(states_data[i])[0] for i in range(model.input_length)]

    frame_counter = 0
    last_prediction = None

state_history = [] # for input_length >= 1, defined in reset()
reset()
first_state = state
last_prediction: torch.Tensor|None = None

INV_KEY_MAP = {v:k for k,v in KEY_MAP.items()}

if __name__ == "__main__":
    while True:
        img = state_to_img(state)
        cv2.imshow(win_name, img)

        key = cv2.waitKey(1000//FPS) & 0xFF

        action = INV_KEY_MAP.get(key, DEFAULT_KEY)
        # print(action)

        if key == ord('q'): # debug frame info
            print(f"\nFrame {frame_counter}")

            amount = [int(np.sum(state[i]).item()) for i in range(model.actions)]
            
            print(f"Amount of each class out of {sum(amount)}:") # or just GRID_SIZE[0] * GRID_SIZE[1]
            for cls, data in COLOR_MAP.items():
                print(f"  {data['name']:10} - {amount[cls]}")

            continue

        elif key == ord('y'):
            # print current hidden states
            if last_prediction is None:
                print("This is the initial state, no predictions yet.")
            else:
                hid_channels: torch.Tensor = last_prediction[0, model.vis_channels:] # (hid_channels, H, W)

                print(f"\nHidden channels shape: {hid_channels.shape}")

                # get values
                with torch.no_grad():
                    hidden = hid_channels.cpu().numpy()
                
                print(f"Mean: {hidden.mean():.3f}, Std: {hidden.std():.3f}")
            
            continue

        elif key == ord('r'): # reset
            reset()
            continue

        elif key == 27: # esc
            break
        elif cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1: # closed window
            break

        if action is not None:
            last_prediction, state = manage_actions(action, state_history, snap_colors)
            frame_counter += 1

    cv2.destroyAllWindows()
