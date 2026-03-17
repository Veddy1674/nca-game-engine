import os
import numpy as np
import snake_env as s
from tqdm import tqdm

PATH = 'snake/dev/data/'

os.makedirs(PATH, exist_ok=True)

# used for debugging
def screenshot(grid, filename):
    from PIL import Image as newImage

    data = np.array([[s.color_map[val] for val in row] for row in grid], dtype=np.uint8)
    
    newImage.fromarray(data).save(filename)

def to_one_hot(frame, num_classes=4):
    # frame is (8, 8) with 0-3 values
    shape = frame.shape

    one_hot = np.zeros((num_classes,) + shape, dtype=np.uint8)

    for c in range(num_classes):
        one_hot[c][frame == c] = 1

    return one_hot # (4, 8, 8)

def format(n): # 100000 -> 100.000 (sorry americans)
    return f"{n:,}".replace(',', '.')

# main
if __name__ == '__main__':
    import time

    t = time.time()

    env = s.SnakeEnv(size=8, seed=51)
    STEPS = 400
    RUN_TIMES = 500 # a .npz for each run
    BEST_MOVE_CHANCE = 0.45 # change of doing the algorithmically best move instead of a random
    CAN_DIE = False # wheter the env is resetted when the snake hits the border
    
    body_count_limit = 6 #(env.size * env.size) - 2

    for run_it in tqdm(range(RUN_TIMES)):
        env.reset()

        frames = []
        actions = []

        bestMoves = 0
        randomMoves = 0

        for i in range(STEPS + 1):
            # (BEST_MOVE_CHANCE * 100)% best move, ((1 - BEST_MOVE_CHANCE) * 100)% random move
            action = 0
            if np.random.rand() < BEST_MOVE_CHANCE:
                action = s.bestMove(env)
                bestMoves += 1
            else:
                action = s.randomMove()
                randomMoves += 1

            # save (state, input), AI will predict (state+1)
            frames.append(to_one_hot(np.array(env.get_state())))
            actions.append(action)
            
            if CAN_DIE:
                env.update(action)
            else:
                env.step(action)

                # count snake body
                body_count = len(env.snake) - 1 # exclude head

                if body_count >= body_count_limit:
                    tqdm.write(f"Aborted (run_it: {run_it}, step: {i}, body count: {body_count})")
                    break # abort
        
        frames.append(to_one_hot(np.array(env.get_state()))) # the last s+1

        # save
        np.savez_compressed(f'{PATH}snake_{run_it}.npz', states=frames, actions=actions)

    print(f'\nDone in {(time.time() - t):.2f}s ({RUN_TIMES} runs, each {STEPS} steps)')