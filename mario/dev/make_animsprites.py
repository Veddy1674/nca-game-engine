import numpy as np
from make_sprites import extract_sprites

PATH = 'mario/dev/'
SPRITE_SIZE = 16

# extract unique sprites and assign a unique color
sprites, colors, _ = extract_sprites(f"{PATH}qmarkblocks.png", SPRITE_SIZE)

print(f"Unique sprites found: {len(sprites)}")
np.savez_compressed(f"{PATH}animsprites{SPRITE_SIZE}x{SPRITE_SIZE}.npz", sprites=sprites, colors=colors)

print("Sprites shape:", sprites.shape) # e.g: (3, 16, 16, 3) means 3 RGB 16x16 sprites
print("Colors shape:", colors.shape) # e.g: (3, 3) means 3 RGB colors