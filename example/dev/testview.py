import cv2
import numpy as np
from env import ExampleEnv, BGR_COLORMAP

SIZE = (8, 8) # W, H
WIN_SIZE = (800, 800) # W, H

KEY_MAP = {
    ord('w'): 0,
    ord('s'): 1,
    ord('a'): 2,
    ord('d'): 3,
}

def render():
    img = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)

    for cls_idx, color in BGR_COLORMAP.items():
        mask = env.state[cls_idx] == 1.0
        img[mask] = color

    img = cv2.resize(img, WIN_SIZE, interpolation=cv2.INTER_NEAREST)

    return img

env = ExampleEnv(*SIZE)

while True:
    cv2.imshow("Example Env", render())
    key = cv2.waitKey(0) & 0xFF # 0 fps (waits for input)

    if key == 27: # esc
        break

    elif key == ord('r'):
        env.reset()
        continue

    elif key in KEY_MAP:
        env.step(KEY_MAP[key])

cv2.destroyAllWindows()