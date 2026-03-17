import numpy as np
from PIL import Image

SPRITES_DATA = np.load("mario/dev/sprites16x16.npz")
GAME_SPRITES = SPRITES_DATA['sprites']
GAME_COLORS = SPRITES_DATA['colors']

ANIMSPRITES_DATA = np.load("mario/dev/anim16x16.npz")
ANIM_GAME_SPRITES = ANIMSPRITES_DATA['sprites']
ANIM_GAME_COLORS = ANIMSPRITES_DATA['colors']

# invert first and last indexes, so that sky is always the last (considered transparency)
GAME_SPRITES[[0, -1]] = GAME_SPRITES[[-1, 0]]
GAME_COLORS[[0, -1]] = GAME_COLORS[[-1, 0]]

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
    def __init__(self, map_path: str, max_widths: list[int]):
        # shape 34, 15, 211 (C, H, W)
        map_array = np.array(Image.open(map_path).convert('RGB'))
        self.map_array = rgb_to_onehot(map_array)

        self.max_widths = max_widths # width of each level (yes, hardcoded)

        levels: int = self.map_array.shape[1] // 15
        print(f"{levels} Levels found.")

        self.max_level = levels
        self.set_level(0) # 0 to max_level - 1

        self.reset()
    
    def reset(self):
        # shape 34, 15, 15 (C, H, W)
        self.cameraX = 0
        self._update_state()
        
        return self.state
    
    def _update_state(self):
        self.state = self.map_array[:, self.level*15:(self.level+1)*15, self.cameraX:self.cameraX+15]
        
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
    
    def set_level(self, level):
        if level < 0 or level >= self.max_level:
            raise ValueError("Level must be between 0 and max_level - 1")
        
        self.level = level
        self.maxCamX = self.max_widths[level] - 15 # 15 is camera width
        # minCamX is always 0
        return self.reset()

    def randomize_level(self):
        self.level = np.random.randint(0, self.max_level)
        return self.reset()

# make dataset
if __name__ == "__main__":
    from tqdm import tqdm
    from time import time

    EPISODES_RANDOM = 600 # episodes where camX and level are randomized
    STEPS = 400 # per episode

    env = MarioEnv(map_path="mario/dev/maps_small.png", max_widths=[211, 164, 213, 238, 159, 212, 213, 164, 192, 237, 389, 229, 229])

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
                state = env.randomize_level()
                # state = env.randomize_camx()

            action = None
            streak = 0

            for _ in range(STEPS):
                states.append(state.copy())
                
                # do rand action and append
                # action, streak = random_action(action, streak)
                # action = np.random.choice([0, 1], p=[0.3, 0.7])
                action = 1
                actions.append(action)

                state = env.step(action)

                if env.cameraX >= env.maxCamX - 1:
                    # tqdm.write(f"Episode {episode}: Early end")
                    break

            states.append(state.copy())

            states = np.array(states)
            actions = np.array(actions)
            
            map_id = np.zeros((states.shape[0], 13, 15), dtype=np.float32)
            map_id[:, env.level, :] = 1.0  # one-hot on level index

            np.savez_compressed(f"mario/dev/data/mario_{episode}.npz", states=states, actions=actions, map_id=map_id)

        if debug:
            # just for debugging
            print(f"\nStates shape: {states.shape}")
            print(f"Actions shape: {actions.shape}")
            print(f"Map id shape: {map_id.shape}")
    
    train(0, EPISODES_RANDOM, debug=True)

    print(f"\nDone in {time() - t:.2f}s.")