import random
import numpy as np

class SnakeEnv:
    def __init__(self, size=4, seed=None):
        self.size = size
        self.rng_seed = seed
        # self.rng = np.random.default_rng(self.rng_seed)
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.rng_seed)
        self.snake = [(0, 0)]
        self.apple = (2, 2)
        self.done = False

    def step(self, action):
        head = self.snake[0]

        # TODO make this a dict like color_map?
        if action == 0: new_head = (head[0] - 1, head[1]) # up
        elif action == 1: new_head = (head[0] + 1, head[1]) # down
        elif action == 2: new_head = (head[0], head[1] - 1) # left
        elif action == 3: new_head = (head[0], head[1] + 1) # right

        # collision
        if not (0 <= new_head[0] < self.size and 0 <= new_head[1] < self.size):
            self.done = True
            return

        self.snake.insert(0, new_head)

        # eat apple
        if new_head == self.apple:
            # regen apple
            all_positions = [(r, c) for r in range(self.size) for c in range(self.size)]
            free_positions = [pos for pos in all_positions if pos not in self.snake] # free position that aren't snake
            
            # choose free space or win
            if free_positions:
                idx = self.rng.integers(len(free_positions))
                self.apple = free_positions[idx]
            else:
                self.done = True
        else:
            self.snake.pop()

    def get_state(self):
        # 0 = snake body (multiple), 1 = snake head (1), 2 = apple (1), 3 = background (multiple)
        grid = [[3 for _ in range(self.size)] for _ in range(self.size)]

        grid[self.apple[0]][self.apple[1]] = 2 # apple

        for s in self.snake:
            grid[s[0]][s[1]] = 0

        head = self.snake[0]
        grid[head[0]][head[1]] = 1

        return grid
    
    def update(self, action): # classic logic of step and reset if done
        self.step(action)

        if self.done:
            self.reset()


def bestMove(env): # returns action
    head = env.snake[0]
    apple = env.apple
    
    # return the best move without having to simulate all 4 moves
    moves = [
        # where each move causes the head to end up
        (head[0] - 1, head[1]), # up
        (head[0] + 1, head[1]), # down
        (head[0], head[1] - 1), # left
        (head[0], head[1] + 1)  # right
    ]

    best_action = 0
    min_dist = float('inf')

    for i, next_pos in enumerate(moves):
        # skip invalid moves
        if not (0 <= next_pos[0] < env.size and 0 <= next_pos[1] < env.size):
            continue
        
        # skip moves which collide with the body
        # if next_pos in env.snake:
        #     continue

        # calc Manhattan distance
        dist = abs(next_pos[0] - apple[0]) + abs(next_pos[1] - apple[1])
        
        if dist < min_dist:
            min_dist = dist
            best_actions = [i] # new best distance

        elif dist == min_dist:
            best_actions.append(i) # same distance as the previous best (add to a list which chooses a random best move)
            
    return random.choice(best_actions) if best_actions else 0

def randomMove(): # returns action
    return random.randint(0, 3)

# RGB
color_map = {
    # same colors as in test.py
    0: [0, 255, 0], # body
    1: [0, 154, 0], # head
    2: [255, 0, 0], # apple
    3: [30, 30, 30] # background
}

bgr_colormap = {k: v[::-1] for k, v in color_map.items()}