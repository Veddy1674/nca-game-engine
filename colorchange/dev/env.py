import numpy as np

class SimpleColorEnv:
    def __init__(self, width=32, height=32, colors=[
            [255, 255, 255],
            [0, 255, 0],
            [255, 0, 0],
            [0, 0, 255]
        ], use_rgb=False):

        self.colors = colors
        self.use_rgb = use_rgb

        if use_rgb:
            self.state = np.zeros((3, height, width), dtype=np.float32) # RGB
        else:
            self.state = np.zeros((len(colors), height, width), dtype=np.float32) # one-hot

        self.idx = 0

    def reset(self):
        self.idx = 0
        return self._updateColor() # set to white

    def step(self):
        self.idx = (self.idx + 1) % len(self.colors)
        return self._updateColor()
    
    def _updateColor(self):
        self.state.fill(0) # set all channels to 0

        if self.use_rgb:
            # set active color (normalized 0-1)
            self.state[:, :, :] = np.array(self.colors[self.idx], dtype=np.float32).reshape(3, 1, 1) / 255.0
        else:
            # set active channel to 1.0!
            self.state[self.idx, :, :] = 1.0
        
        return self.state.copy()
    
    def get_state(self):
        return self.state