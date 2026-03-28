CONFIG_FILE = "sand/config.py"
MODEL_PATH = "sand/sand.pt"

# ignore all of this below
from NACE import NACE
import torch

model: NACE = None
optimizer: torch.optim.Optimizer = None
loss_calc: torch.nn.Module = None
STEPS: int = None
BATCH_SIZE: int = None
LOG_SEGMENTS: int = None
# LOAD_MODEL: str = None
FILE_NAME: str = None
DATA_GLOB: str = None
MICROSTEPS: int = None
TRAIN_STEPS: int = None
POOL_LENGTH: int = None
LOAD_QUICK: bool = None
LOAD_INSTANT: bool = None
EXTRA_MAPS: dict = None
LOSS_GRAPH: str = None
LOAD_OPTIMIZER: bool = None

FIRST_DATA_FILE: str = None
GRID_SIZE: tuple[int, int] = None
STARTING_IMAGE: str = None
BIT_DEPTH_LEVELS: int = None
COLOR_MAP: dict = None
bgr_colormap: dict = None
TEMPERATURE: float = None
TOP_P: float = None

KEY_MAP: dict = None
DEFAULT_KEY: int = None
FPS: int = None
# FPS_PYGAME: int = None
# REFRESH_RATE: int = None
# VSYNC: bool = None
# HIDE_INFO: bool = None
WIN_SIZE: tuple[int, int] = None

if __name__ == "__main__":
    print("This file is not meant to be run directly.")