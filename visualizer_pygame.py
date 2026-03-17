import pygame
import numpy as np

import visualizer
from visualizer import *

REFRESH_RATE = globals().get('REFRESH_RATE', 100) # adapt to the monitor's refresh rate
VSYNC = globals().get('VSYNC', True)
HIDE_INFO = globals().get('HIDE_INFO', False) # hides fps and "paused" texts

if DEFAULT_KEY is None:
    print("DEFAULT_KEY Should be initialized in the config file for this script to work properly.")
    exit()

def to_surface(state: np.ndarray) -> pygame.Surface:
    img = post_processing(state_to_img(state))
    img_rgb = img[:, :, ::-1].transpose(1, 0, 2)
    return pygame.transform.scale(pygame.surfarray.make_surface(img_rgb), WIN_SIZE)

CONTINUOUS_KEYS = [getattr(pygame, f'K_{chr(key)}') if chr(key).isalpha() else 
                   pygame.K_SPACE if key == ord(' ') else 
                   key for key in KEY_MAP.values()]

PG_TO_ACTION = {
    getattr(pygame, f'K_{chr(key)}') if chr(key).isalpha() else 
    pygame.K_SPACE if key == ord(' ') else 
    key: action_idx 
    for action_idx, key in KEY_MAP.items()
}
SPECIAL = {pygame.K_r, pygame.K_q, pygame.K_y, pygame.K_p}

pygame.init()
screen = pygame.display.set_mode(WIN_SIZE, vsync=(1 if VSYNC else 0))
pygame.display.set_caption("NCA Visualizer | P = start/pause")
clock = pygame.time.Clock()
pygame.key.set_repeat(0)

paused = False#True
last_step_time = 0
STEP_INTERVAL = 1000 // (FPS if FPS is not None else globals().get('FPS_PYGAME', 60))

fpsFont = pygame.font.SysFont(None, 36)

running = True
while running:
    fired = set()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:

            match event.key:
                case pygame.K_ESCAPE:
                    running = False
                case pygame.K_p:
                    paused = not paused
                case pygame.K_r:
                    resetAll()
                case pygame.K_q:
                    print(f"\nFrame {visualizer.frame_counter}")

                    amount = [int(np.sum(visualizer.state[i]).item()) for i in range(model.actions)]
                    print(f"Amount of each class out of {sum(amount)}:")

                    for cls, data in COLOR_MAP.items():
                        print(f"  {data['name']:10} - {amount[cls]}")

                case pygame.K_y:
                    lp = visualizer.last_prediction

                    if lp is None:
                        print("Initial state — no predictions yet.")
                    else:
                        hid = lp[0, model.vis_channels:].cpu().numpy()
                        print(f"\nHidden channels: {hid.shape[0]}: mean={hid.mean():.3f}, std={hid.std():.3f}")
                

    now = pygame.time.get_ticks()
    if not paused and (now - last_step_time) >= STEP_INTERVAL:
        pressed = pygame.key.get_pressed()
        action = DEFAULT_KEY
        for pg_key, act in PG_TO_ACTION.items():
            if pressed[pg_key]:
                action = act
                break

        visualizer.last_prediction, next_frame = manage_actions(action, visualizer.state_history, snap_colors, predict_next, apply_top_p)
        if next_frame is not None:
            visualizer.state = next_frame
            visualizer.frame_counter += 1

        last_step_time = now

    # draw
    screen.blit(to_surface(visualizer.state), (0, 0))

    if not HIDE_INFO:
        if not paused:
            screen.blit(fpsFont.render(f"FPS: {clock.get_fps():.0f}", True, (220, 220, 220)), (8, 8))
        else:
            screen.blit(fpsFont.render("PAUSED", True, (220, 220, 220)), (8, 8))

    pygame.display.flip()

    clock.tick(REFRESH_RATE)

pygame.quit()