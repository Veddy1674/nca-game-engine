import cv2
import numpy as np
from snake_env import SnakeEnv, bestMove, bgr_colormap

SIZE = 8
CELL = 80
FPS = 8

KEY_MAP = {
    ord('w'): 0,
    ord('s'): 1,
    ord('a'): 2,
    ord('d'): 3,
}

def render(env, score):
    h = w = SIZE * CELL
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    grid = env.get_state()
    for r in range(SIZE):
        for c in range(SIZE):
            color = bgr_colormap[grid[r][c]]
            y1, y2 = r * CELL, (r + 1) * CELL
            x1, x2 = c * CELL, (c + 1) * CELL
            frame[y1:y2, x1:x2] = color
            cv2.rectangle(frame, (x1, y1), (x2-1, y2-1), (20, 20, 20), 1)

    cv2.putText(frame, f"Score: {score}  WASD=move  B=best  R=reset  Q=quit",
                (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return frame

env = SnakeEnv(size=SIZE, seed=51)
score = 0
action = 3

cv2.namedWindow("Snake", cv2.WINDOW_AUTOSIZE)

while True:
    if env.done:
        env.reset()
        score = 0
        action = 3

    cv2.imshow("Snake", render(env, score))
    key = cv2.waitKey(1000 // FPS) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r'):
        env.reset()
        score = 0
        action = 3
        continue
    elif key == ord('b'):
        action = bestMove(env)
    elif key in KEY_MAP:
        action = KEY_MAP[key]

    prev_len = len(env.snake)
    env.step(action)
    if len(env.snake) > prev_len:
        score += 1

cv2.destroyAllWindows()