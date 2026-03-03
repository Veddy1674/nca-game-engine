import numpy as np
from PIL import Image as newImage

COLOR_MAP = { # RGB
    0: [22, 22, 22],
    1: [119, 119, 119],
    2: [193, 171, 73],
    3: [38, 38, 38]
}

id = 0
filename = f"sand/dev/data/sand_0_51744.npz"

with np.load(filename) as data:
    states = data['states']
    
    for idx in [id, id+1]:
        state = states[idx]
        class_map = np.argmax(state, axis=0)
        
        rgb = np.zeros((48, 48, 3), dtype=np.uint8)
        for val, color in COLOR_MAP.items():
            rgb[class_map == val] = color
            
        newImage.fromarray(rgb).save(f"sand/dev/data/state_{idx}.png")
    
    print(f'Action: {data["actions"][id]}')