import importlib.util, sys, os

def load_configuration():

    # arg0 is configPath, if null ask via input()
    if len(sys.argv) < 2:
        try:
            configPath = input("Config Path: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            sys.exit()
    else:
        configPath = sys.argv[1]
    
    if not os.path.exists(configPath):
        print(f"File not found: {configPath}")
        sys.exit()
    
    spec = importlib.util.spec_from_file_location("main", configPath)
    if spec is None:
        print(f"Invalid module: {configPath}")
        sys.exit()
    
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    import inspect
    caller_globals = inspect.currentframe().f_back.f_globals
    
    # only export variables defined below, EXCLUDE __name__, and such, which cause issues
    config_vars = {
        k: v for k, v in vars(cfg).items() 
        if not k.startswith('__')
    }
    caller_globals.update(config_vars)
    
    # return cfg

# type hints (found in every configuration):
from NACE import NACE
import torch

model: NACE
optimizer: torch.optim.Optimizer
loss_calc: torch.nn.Module
STEPS: int # how many steps (not epochs) the AI model is trained for
BATCH_SIZE: int # allows training on multiple inputs/targets at the same time, increases VRAM usage
LOG_SEGMENTS: int # amount of info-prints total during training, (e.g: 10 log segments on 1000 steps means info will be printed each 1000/10 = 100 steps)
# reccomended values are 10 or 100, higher values means more precision on loss graphs

# LOAD_MODEL: str # path of the model loaded before training
FILE_NAME: str # path of the model saved after training
DATA_GLOB: str # path of data to load, use asteriskes to include multiple files
MICROSTEPS: int # how many pixels can information propagate to
TRAIN_STEPS: int # (unstable) trains TRAIN_STEPS steps in a row each iteration, resulting in the AI less likely to accumulate errors over time
POOL_LENGTH: int # (unstable) trains randomly the AI on states predicted by itself, resulting in the AI more likely to be able to fix its own mistakes
LOAD_QUICK: bool # wheter to load everything to RAM (for small datasets) or use lazy loading
LOAD_INSTANT: bool # wheter to load everything to RAM/VRAM (for tiny datasets where CPU loading data is the bottleneck)
EXTRA_MAPS: dict # extra data besides states and actions (e.g: {'level': 'long'}), must be fed to input even during inference
LOSS_GRAPH: str # path where to save the loss graph (or None)
LOAD_OPTIMIZER: bool # if False uses the lr defined in model, otherwise continues from the loaded model's lr

MODEL_PATH: str # path of the model used for the visualizer (can be the same as LOAD_MODEL or FILE_NAME)
FIRST_DATA_FILE: str # path of a single data file used for the visualizer (can be None if using STARTING_IMAGE)
GRID_SIZE: tuple[int, int] # size of the grid (H, W), used for the visualizer mostly
STARTING_IMAGE: str # path of an image used for the visualizer (can be None if using FIRST_DATA_FILE), repeated for each input_length for simplicity
BIT_DEPTH_LEVELS: int # (only if COLOR_MAP is None, thus using RGB) quantize colors, avoids the AI making colors extremely bright/dark due to small errors accumulation
COLOR_MAP: dict # None if using RGB, otherwise a dict containing 'name' and 'color' (RGB) for each "object"
bgr_colormap: dict # auto-generated
TEMPERATURE: float # chance of the model picking less likely options instead of always the safest, only applied if > 1.0
TOP_P: float # model will only pick among the most likely options whose cumulative probability exceeds TOP_P, only applied if temperature > 1.0

KEY_MAP: dict # maps action indexes to pygame key codes, (e.g: {0: ord('w'), 1: ord('s'), 2: ord('a'), 3: ord('d')})
DEFAULT_KEY: int # action forwarded to the model when no key is pressed
FPS: int # frames per second for the visualizer (if action is none then DEFAULT_KEY will be used), if this parameter is None, it will wait for user input each step
# FPS_PYGAME: int
# REFRESH_RATE: int
# VSYNC: bool
# HIDE_INFO: bool
WIN_SIZE: tuple[int, int] # auto-calculated based on GRID_SIZE and base_size defined in the config file