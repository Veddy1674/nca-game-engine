import numpy as np

# RGB
COLOR_MAP = {
    0: [240, 240, 240], # white (player)
    1: [33, 33, 33], # background
}

BGR_COLORMAP = {k: v[::-1] for k, v in COLOR_MAP.items()}

class ExampleEnv:
    def __init__(self, width=8, height=8):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.state = np.zeros((2, self.height, self.width), dtype=np.float32)

        self.state[0, self.width // 2, self.height // 2] = 1.0 # player

        # set background where there isn't player
        mask = self.state[0] == 0.0
        self.state[1, mask] = 1.0 # background

        return self.state

    def step(self, action):
        shift = 1
        axis = 1

        if action == 0: # up
            shift = -1
        elif action == 1: # down
            pass
        elif action == 2: # left
            shift = -1
            axis = 2
        elif action == 3: # right
            axis = 2
        
        self.state = np.roll(self.state, shift=shift, axis=axis)
        
        return self.state

if __name__ == "__main__":
    from tqdm import tqdm

    def rand_action():
        return np.random.randint(0, 4)

    env = ExampleEnv(width=8, height=8)
    s = env.reset()

    states = []
    actions = []

    STEPS = 10_000

    for _ in tqdm(range(1, STEPS)):
        states.append(s.copy())

        a = rand_action()
        s = env.step(action=a)

        actions.append(a)
    
    states.append(s.copy())
    
    states = np.array(states)
    actions = np.array(actions)

    np.savez("example/dev/data/example_0.npz", states=states, actions=actions)

    print("States shape:", states.shape)
    print("Actions shape:", actions.shape)