import pygame
import numpy as np
import visualizer
from visualizer import *

REFRESH_RATE = 100 # adapt to the monitor's refresh rate

def to_surface(state: np.ndarray) -> pygame.Surface:
    img = state_to_img(state)
    img_rgb = img[:, :, ::-1].transpose(1, 0, 2)
    return pygame.transform.scale(pygame.surfarray.make_surface(img_rgb), WIN_SIZE)

CONTINUOUS_KEYS = [pygame.K_a, pygame.K_d, pygame.K_SPACE]
PG_TO_ACTION = {
    pygame.K_a:     INV_KEY_MAP.get(ord('a')),
    pygame.K_d:     INV_KEY_MAP.get(ord('d')),
    pygame.K_SPACE: INV_KEY_MAP.get(ord(' ')),
}
SPECIAL = {pygame.K_r, pygame.K_q, pygame.K_y, pygame.K_p}

pygame.init()
screen = pygame.display.set_mode(WIN_SIZE)
pygame.display.set_caption("NCA Visualizer  |  P = start/pause")
clock = pygame.time.Clock()
pygame.key.set_repeat(0)

paused = True
held_specials = set()
last_step_time = 0
STEP_INTERVAL = 1000 // FPS  # model steps at same rate as cv2.waitKey(1000//FPS)

running = True
while running:
    fired = set()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key in SPECIAL and event.key not in held_specials:
                fired.add(event.key)
                held_specials.add(event.key)
        elif event.type == pygame.KEYUP:
            held_specials.discard(event.key)

    if pygame.K_p in fired:
        paused = not paused

    if not paused:
        if pygame.K_r in fired:
            reset()
        if pygame.K_q in fired:
            s = visualizer.state
            print(f"\nFrame {visualizer.frame_counter}")
            amount = [int(np.sum(s[i]).item()) for i in range(model.actions)]
            print(f"Amount of each class out of {sum(amount)}:")
            for cls, data in COLOR_MAP.items():
                print(f"  {data['name']:10} - {amount[cls]}")
        if pygame.K_y in fired:
            lp = visualizer.last_prediction
            if lp is None:
                print("Initial state — no predictions yet.")
            else:
                hid = lp[0, model.vis_channels:].cpu().numpy()
                print(f"\nHidden channels: {hid.shape}  mean={hid.mean():.3f}  std={hid.std():.3f}")

        now = pygame.time.get_ticks()
        if now - last_step_time >= STEP_INTERVAL:
            pressed = pygame.key.get_pressed()
            action = DEFAULT_KEY
            for pg_key in CONTINUOUS_KEYS:
                if pressed[pg_key]:
                    action = PG_TO_ACTION.get(pg_key, DEFAULT_KEY)
                    break

            next_frame = predict_next(visualizer.state, action)
            next_frame = np.eye(4)[next_frame].transpose(2, 0, 1)
            visualizer.state_history.append(next_frame)
            if len(visualizer.state_history) > model.input_length:
                visualizer.state_history.pop(0)
            visualizer.state = next_frame
            visualizer.frame_counter += 1
            last_step_time = now

    screen.blit(to_surface(visualizer.state), (0, 0))
    if paused:
        font = pygame.font.SysFont(None, 36)
        screen.blit(font.render("PAUSED  —  P to start", True, (220, 220, 220)), (8, 8))
    pygame.display.flip()
    clock.tick(REFRESH_RATE)

pygame.quit()