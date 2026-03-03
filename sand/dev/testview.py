import cv2
import numpy as np
from env import FallingSandEnv

def render(state):
    h, w = state.shape[1], state.shape[2]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i, c in bgr_colors.items():
        mask = state[i] == 1
        img[mask] = c
    
    img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("test", img)

env = FallingSandEnv(width=48, height=48)
state = env.reset()

colors = [
    [22, 22, 22], # floor
    [119, 119, 119], # rectangle (sand spawner)
    [193, 171, 73], # sand
    [38, 38, 38] # background
]

bgr_colors = {i: c[::-1] for i, c in enumerate(colors)}

frame = 0
while True:
    render(state)
    key = cv2.waitKey(16) & 0xFF

    if key == 27:
        break

    action = 0

    if key == ord('d'):
        action = 1 # right

    elif key == ord('a'):
        action = 2 # left

    elif key == ord(' '):
        action = 3

    state = env.step(action)

    frame += 1
    if (frame + 1) % 100 == 0: # debug
        print(f"Frame {frame+1}")

cv2.destroyAllWindows()