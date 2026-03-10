from PIL import Image
import numpy as np

PATH = 'mario/dev/ccc/'
COLOR_MAP_LIST = [
    [0, 0, 0], # black

    # goomba
    [156, 74, 0],
    [255, 206, 197],

    [255, 255, 255], # transparency (in the images represented with white)
]

def rgb_to_onehot(image: np.ndarray) -> np.ndarray:
    h, w, _ = image.shape

    onehot = np.zeros((len(COLOR_MAP_LIST), h, w), dtype=np.uint8)

    for i, color in enumerate(COLOR_MAP_LIST):
        mask = np.all(image == color, axis=-1)
        onehot[i][mask] = 1
    
    return onehot

frame1 = rgb_to_onehot(np.array(Image.open(f"{PATH}frame1.png").convert('RGB')))
frame2 = rgb_to_onehot(np.array(Image.open(f"{PATH}frame2.png").convert('RGB')))

states = np.array([frame1, frame2] * 200)
actions = np.array([0] * (len(states) - 1))

np.savez_compressed(f"{PATH}ccc_0.npz", states=states, actions=actions)

print("States shape:", states.shape)
print("Actions shape:", actions.shape)