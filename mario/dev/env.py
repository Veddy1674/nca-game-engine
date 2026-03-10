import numpy as np
from PIL import Image

SPRITES_DATA = np.load("mario/dev/sprites16x16.npz")
GAME_SPRITES = SPRITES_DATA['sprites']
GAME_COLORS = SPRITES_DATA['colors']

ANIMSPRITES_DATA = np.load("mario/dev/animsprites16x16.npz")
ANIM_GAME_SPRITES = ANIMSPRITES_DATA['sprites']
ANIM_GAME_COLORS = ANIMSPRITES_DATA['colors']

# unique color for each sprite
COLOR_MAP_LIST = GAME_COLORS.tolist()
BGR_COLORS = {i: c[::-1] for i, c in enumerate(COLOR_MAP_LIST)} # for cv2 rendering

# to see which index corresponds to which sprite:
# Image.fromarray(GAME_SPRITES[11]).show()
# exit()

def rgb_to_onehot(image: np.ndarray) -> np.ndarray:
    h, w, _ = image.shape

    onehot = np.zeros((len(COLOR_MAP_LIST), h, w), dtype=np.uint8)

    for i, color in enumerate(COLOR_MAP_LIST):
        mask = np.all(image == color, axis=-1)
        onehot[i][mask] = 1
    
    return onehot

# (used externally), uses BGR_COLORS instead of COLOR_MAP!
def onehot_to_rgb(state: np.ndarray) -> np.ndarray: # (C, H, W)
    h, w = state.shape[1], state.shape[2]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i, color in BGR_COLORS.items():
        mask = state[i] == 1
        img[mask] = color

    return img

class MarioEnv:
    def __init__(self, map_path: str):
        # shape 34, 15, 211 (C, H, W)
        self.map_array = rgb_to_onehot(np.array(Image.open(map_path).convert('RGB')))

        self.maxCamX = self.map_array.shape[2] - 15 # 211-15 = 196
        # minCamX is 0

        self.reset()
    
    def reset(self):
        # shape 34, 15, 15 (C, H, W)
        self.cameraX = 0
        self._update_state()
        
        return self.state
    
    def _update_state(self):
        self.state = self.map_array[:, :, self.cameraX:self.cameraX+15]
        
    def step(self, action):

        if action == 1: # right
            self.cameraX = min(self.cameraX + 1, self.maxCamX) # 211-15 = 196
        elif action == 2: # left
            self.cameraX = max(self.cameraX - 1, 0)
        
        self._update_state()
        return self.state
    
    def randomize_camx(self):
        self.cameraX = np.random.randint(0, self.maxCamX + 1)
        return self.step(0) # update and return state

# make dataset
if __name__ == "__main__":
    from tqdm import tqdm
    from time import time

    EPISODES_STARTAREA = 100 # episodes where camX is 0
    EPISODES_RANDOM = 100 # episodes where camX is randomized
    STEPS = 1000 # per episode

    env = MarioEnv(map_path="mario/dev/map_211x15.png")

    ACTIONS = [0, 1, 2] # no-op, right, left
    PROBS = [0.2, 0.4, 0.4]

    def random_action(prev_action=None, streak=0):
        if prev_action is None or np.random.random() > 0.5 / (2 ** streak):
            return np.random.choice(ACTIONS, p=PROBS), 0
        
        return prev_action, streak + 1
    
    t = time()

    # separate files for each episode
    # (recommended if TRAIN_STEPS in config is > 1, otherwise the NCA could learn to reset the game randomly)
    def train(start_idx: int, end_idx: int, debug=False):
        for episode in tqdm(range(start_idx, start_idx + end_idx), desc=f"{start_idx}-{start_idx + end_idx}"):
            states = []
            actions = []

            state = env.reset()

            if episode > 0: # if isn't first (just a convention, so that first file always begins with camx to 0)
                state = env.randomize_camx()

            action = None
            streak = 0

            for _ in range(STEPS):
                states.append(state.copy())
                
                # do rand action and append
                action, streak = random_action(action, streak)
                actions.append(action)

                state = env.step(action)

            states.append(state.copy())

            states = np.array(states)
            actions = np.array(actions)

            np.savez_compressed(f"mario/dev/data/mario_{episode}.npz", states=states, actions=actions)

        if debug:
            # just for debugging
            print(f"\nStates shape: {states.shape}")
            print(f"Actions shape: {actions.shape}")
    
    train(0, EPISODES_STARTAREA)
    train(EPISODES_STARTAREA, EPISODES_RANDOM, debug=True)

    print(f"\nDone in {time() - t:.2f}s.")