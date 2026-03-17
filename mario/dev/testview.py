import cv2
import numpy as np
from env import MarioEnv, GAME_SPRITES, onehot_to_rgb
from make_sprites import SPRITE_SIZE
from PIL import Image as newImage

env = MarioEnv(map_path="mario/dev/maps_small.png", max_widths=[211, 164, 213, 238, 159, 212, 213, 164, 192, 237, 389, 229, 229])
state = env.reset()

WIN_SIZE = (896, 896) # prefering multiples of SPRITE_SIZE (16)

def render(state: np.ndarray, with_sprites: bool) -> np.ndarray: # (C, H, W)
    if not with_sprites:
        return onehot_to_rgb(state)

    h, w = state.shape[1], state.shape[2]
    img = np.zeros((h * SPRITE_SIZE, w * SPRITE_SIZE, 3), dtype=np.uint8)
    
    for y in range(h):
        for x in range(w):
            color = np.argmax(state[:, y, x])
            
            # get sprite for the color channel
            sprite = GAME_SPRITES[color]
            
            # draw sprite
            img[y*SPRITE_SIZE : (y+1)*SPRITE_SIZE, x*SPRITE_SIZE : (x+1)*SPRITE_SIZE] = sprite
    
    return img[:, :, ::-1]

# render sprites smoothly via states interpolation
def render_smooth(state_a: np.ndarray, state_b: np.ndarray, t: int) -> np.ndarray:
    if (state_b is None) or np.array_equal(state_a, state_b):
        return render(state_a, with_sprites=True)

    # render both states
    img_a = render(state_a, with_sprites=True)
    img_b = render(state_b, with_sprites=True)

    # create new image
    img = np.zeros_like(img_a)
    W = img_a.shape[1]

    at = abs(t)
    offset = SPRITE_SIZE - at # offset of state B
    
    if t > 0:  # right
        img[:, :W - at] = img_a[:, at:]
        img[:, offset:] = img_b[:, :W - offset]
    else:  # left
        img[:, at:] = img_a[:, :W - at]
        img[:, :W - offset] = img_b[:, offset:]
    
    return img

def reset():
    global state, frame, scroll_t, next_state
    state = env.reset()

    frame = 0
    scroll_t = 0 # from 1 to SPRITE_SIZE - 1
    next_state = None

pending_action = None

def step(action):
    global state, next_state, scroll_t, pending_action

    if action == 1:
        scroll_t += SCROLL_SPEED
    elif action == 2:
        scroll_t -= SCROLL_SPEED

    # back to origin - undo using the action that generated next_state
    if scroll_t == 0 and next_state is not None:
        undo = 2 if pending_action == 1 else 1
        env.step(undo)
        next_state = None
        pending_action = None
        return

    # generate next state once when we start moving
    if next_state is None:
        candidate = env.step(action)
        if np.array_equal(candidate, state): # nothing changed
            scroll_t = 0
            return
        next_state = candidate
        pending_action = action

    # snap
    if scroll_t >= SPRITE_SIZE or scroll_t <= -SPRITE_SIZE:
        state = next_state
        next_state = None
        pending_action = None
        scroll_t = 0

SCROLL_SPEED = 16
reset()

while True:
    # print(scroll_t)
    frame += 1
    if (frame) % 100 == 0: # debug
        print(f"Frame {frame}")
    
    # simple snap-to-grid-like rendering
    # img = render(state, with_sprites=True)

    # rendering with interpolation (sprites)
    img = render_smooth(state, next_state, scroll_t)

    img_resized = cv2.resize(img, WIN_SIZE, interpolation=cv2.INTER_NEAREST)

    cv2.imshow("test", img_resized)
    key = cv2.waitKey(16) & 0xFF

    if key == 27:
        break
    elif key == ord('q'): # screenshot
        newImage.fromarray(img[:, :, ::-1]).save(f"mario/dev/screenshot_f{frame}.png")
        continue
    elif key == ord('r'): # reset
        reset()
        continue
    elif key == ord('y'):
        # print info about env
        print(f'Camera X: {env.cameraX}')

    if key == ord('d'): # right
        step(action=1)

    elif key == ord('a'): # left
        step(action=2)

    elif key == ord('\\'):
        step(action=0)
    
    elif key == ord('1'):
        env.set_level((env.level + 1) % env.max_level)
        reset()
        print("Level", env.level + 1)

cv2.destroyAllWindows()