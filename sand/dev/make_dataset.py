import numpy as np
from env import FallingSandEnv
from tqdm import tqdm
from time import time
from multiprocessing import Process
import os

SIZE = (48, 48) # W, H
EPISODES = 100
STEPS = 600
PATH = "sand/dev/data"
FILENAME = "sand"
THREADS = 14 # depends on your CPU!

def run_episode(ep):
    env = FallingSandEnv(*SIZE, sand_gradient=False)

    states = []
    actions = []

    state = env.reset()
    action = env.random_action()
    streak = 0

    for _ in range(STEPS):
        states.append(state.copy())
        actions.append(action)

        state = env.step(action)

        new_action = env.random_action(prev_action=action, streak=streak)

        streak = (streak + 1) if new_action == action else 0

        action = new_action

    states.append(state.copy()) # last frame

    states = np.array(states)
    actions = np.array(actions)

    np.savez_compressed(f"{PATH}/{FILENAME}_{ep}_{os.getpid()}.npz", states=states, actions=actions)
    return ep

if __name__ == '__main__':
    os.makedirs(PATH, exist_ok=True)

    t = time()

    # parallelization
    processes: list[Process] = []
    for ep in tqdm(range(EPISODES), desc="Creating dataset"):
        p = Process(target=run_episode, args=(ep,))
        p.start()

        processes.append(p)

        if len(processes) >= THREADS:
            processes[0].join()
            processes.pop(0)

    # wait for the rest
    for p in processes:
        p.join()

    print(f"Done in {time() - t:.2f}s.")