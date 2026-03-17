from PIL import Image, ImageDraw
import hashlib, numpy as np

def representative_grid(input_path: str, output_path: str, color):
    img = Image.open(input_path).convert('RGB')
    
    new_width = img.width + (img.width // SPRITE_SIZE - 1)
    new_height = img.height + (img.height // SPRITE_SIZE - 1)
    
    new_img = Image.new('RGB', (new_width, new_height), color='black')
    
    # copy pixels with gaps
    for y in range(img.height):
        for x in range(img.width):
            new_x = x + (x // SPRITE_SIZE)
            new_y = y + (y // SPRITE_SIZE)
            new_img.putpixel((new_x, new_y), img.getpixel((x, y)))
    
    # draw grid lines:
    draw = ImageDraw.Draw(new_img)
    
    # vertical lines
    for i in range(1, img.width // SPRITE_SIZE):
        x = i * SPRITE_SIZE + i - 1
        draw.line([(x, 0), (x, new_height)], fill=color, width=1)
    
    # horizontal lines
    for i in range(1, img.height // SPRITE_SIZE):
        y = i * SPRITE_SIZE + i - 1
        draw.line([(0, y), (new_width, y)], fill=color, width=1)
    
    new_img.save(output_path)

# extract sprites from image and assign unique colors (average)
def extract_sprites(input_path: str, SPRITE_SIZE: int):
    img = np.array(Image.open(input_path).convert('RGB'))
    
    h, w = img.shape[:2]
    sprites_x = w // SPRITE_SIZE
    sprites_y = h // SPRITE_SIZE
    
    sprite_map = {} # hash to index

    sprites = []
    colors = []
    used_colors = set()

    # output color map (one pixel per sprite)
    color_map = np.zeros((sprites_y, sprites_x, 3), dtype=np.uint8)
    
    for y in range(sprites_y):
        for x in range(sprites_x):
            # extract sprite (with numpy it's faster)
            sprite = img[y*SPRITE_SIZE : (y+1)*SPRITE_SIZE, x*SPRITE_SIZE : (x+1)*SPRITE_SIZE]
            
            # hash content
            hash = hashlib.md5(sprite.tobytes()).digest()
            
            if hash not in sprite_map:
                sprite_map[hash] = len(sprites)
                sprites.append(sprite)
                
                # calc average color
                avg_color = sprite.mean(axis=(0, 1)).astype(np.uint8)
                avg_tuple = tuple(avg_color)
                
                # if not unique, add +1 to green (convention)
                while avg_tuple in used_colors:
                    avg_tuple = (avg_tuple[0], (avg_tuple[1] + 1) & 0xFF, avg_tuple[2]) # & 0xFF avoids overflow!
                
                used_colors.add(avg_tuple)
                colors.append(list(avg_tuple))
            
            color_map[y, x] = colors[sprite_map[hash]] # set chosen color for this sprite
    
    return np.array(sprites), np.array(colors), color_map

PATH = 'mario/dev/'
SPRITE_SIZE = 16

if __name__ == "__main__":
    # this is used for illustrative purposes only
    # representative_grid(f"{PATH}allmaps.png", f"{PATH}maps_grid.png", color=(33, 33, 33))

    # extract unique sprites and assign a unique color
    sprites, colors, color_map = extract_sprites(f"{PATH}allmaps.png", SPRITE_SIZE)

    print(f"Unique sprites found: {len(sprites)}")
    np.savez_compressed(f"{PATH}sprites{SPRITE_SIZE}x{SPRITE_SIZE}.npz", sprites=sprites, colors=colors)

    print("Sprites shape:", sprites.shape) # e.g: (36, 16, 16, 3) means 36 RGB 16x16 sprites
    print("Colors shape:", colors.shape) # e.g: (36, 3) means 36 RGB colors

    # each pixel rapresents one sprite
    Image.fromarray(color_map).save(f"{PATH}maps_small.png")