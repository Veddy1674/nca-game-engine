import numpy as np

class FallingSandEnv:
    def __init__(self, width=96, height=96):

        self.width = width
        self.height = height

        self.state = self.reset()

    def reset(self):
        self.recX = 8

        self.state = np.zeros((4, self.height, self.width), dtype=np.float32) # empty
        
        # floor 8 pixels from bottom
        self.state[0, -8:, :] = 1.0
        self._updateAll()

        return self.state

    def step(self, action):
        limitX = 0 # a higher limitX confuses the NCA

        if action == 1: # right
            self.recX = min(self.recX + 1, self.width - 10-limitX)

        elif action == 2: # left
            self.recX = max(self.recX - 1, 0+limitX)

        elif action == 3: # spawn sand
            self._spawnSand()

        # action 0 is noop
        self._updateAll()

        return self.state
    
    def _updateRec(self):
        self.state[1, :, :] = 0.0 # clear old position

        # rec (10x8)
        self.state[1, 0:8, self.recX:self.recX + 10] = 1.0
    
    def _updateBackground(self):
        # background on empty spaces
        self.state[3] = 1.0 - np.max(self.state[:3], axis=0)

    def _updateAll(self):
        # floor is static

        self._updateRec()
        self._updateSand()
        self._updateBackground()
    
    def _spawnSand(self):
        # deterministic pattern
        pattern = np.array([
            [0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
            [0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
        ], dtype=np.float32)

        h, w = pattern.shape

        # 7:7 so that it's attached to the rectangle
        self.state[2, 7:7+h, self.recX:self.recX+w] = pattern
    
    """
    def _updateSand(self):
        sand = self.state[2]
        newSand = np.zeros_like(sand)

        for y in reversed(range(self.height-1)):
            for x in range(self.width):

                if sand[y, x] == 1:
                    # if there is nothing below
                    if sand[y+1, x] == 0 and self.state[0, y+1, x] == 0:
                        newSand[y+1, x] = 1
                    else:
                        # diagonals
                        moved = False
                        for dx in [-1,1]:
                            nx = x + dx
                            if 0 <= nx < self.width and sand[y+1, nx] == 0 and self.state[0, y+1, nx] == 0:
                                newSand[y+1, nx] = 1
                                moved = True
                                break
                        
                        if not moved:
                            newSand[y, x] = 1

        self.state[2] = newSand
    """

    # optimized
    def _updateSand(self):
        sand = self.state[2]
        newSand = np.zeros_like(sand)

        ys, xs = np.where(sand == 1)

        # reorder for y increasing
        order = np.argsort(-ys)
        ys, xs = ys[order], xs[order]

        for y, x in zip(ys, xs):
            # bottom is empty
            if sand[y+1, x] == 0 and self.state[0, y+1, x] == 0:
                newSand[y+1, x] = 1
            else:
                moved = False
                for dx in [-1,1]:
                    nx = x + dx
                    if 0 <= nx < self.width and sand[y+1, nx] == 0 and self.state[0, y+1, nx] == 0:
                        newSand[y+1, nx] = 1
                        moved = True
                        break
                if not moved:
                    newSand[y, x] = 1

        self.state[2] = newSand
    
    def get_state(self):
        return self.state
    
    @staticmethod
    def random_action(prev_action=None, streak=0):
        # 30% left 30% right 20% nothing 10% spawn
        base = {
            0: 0.20, # nothing
            1: 0.30, # right
            2: 0.30, # left
            3: 0.10, # spawn
        }

        # chance of doing the previous action (decreases with streak, to ~0 in 6 streaks)
        if prev_action is not None and prev_action != 0:
            boost = max(0.5 - streak * 0.08, 0.0)
            base[prev_action] += boost

        actions = list(base.keys())
        probs = np.array(list(base.values()))
        
        probs /= probs.sum()
        return np.random.choice(actions, p=probs)