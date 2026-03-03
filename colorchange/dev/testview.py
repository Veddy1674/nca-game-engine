import cv2
import numpy as np
from env import SimpleColorEnv

def render(state):
    h, w = state.shape[1], state.shape[2]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i, color in bgr_colors.items():
        mask = state[i] == 1
        img[mask] = color
    
    img = cv2.resize(img, (600, 600), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("test", img)

env = SimpleColorEnv()
state = env.reset()

bgr_colors = {i: color[::-1] for i, color in enumerate(env.colors)}

while True:
    render(state)
    key = cv2.waitKey(100) & 0xFF

    if key == 27:
        break

    if key == ord(' '):
        state = env.step()

cv2.destroyAllWindows()