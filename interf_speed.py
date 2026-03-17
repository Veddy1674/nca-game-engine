import time
import numpy as np
import torch
from mario.configMapRenderer import *

def test_inference_speed(model, num_runs=500, warmup=50):
    device = model.device
    
    state = np.zeros((model.vis_channels, GRID_SIZE[0], GRID_SIZE[1]), dtype=np.float32)
    state[-1] = 1.0
    
    state_history = [state.copy() for _ in range(model.input_length)]
    
    for _ in range(warmup):
        action = np.random.randint(0, model.actions) if model.actions > 1 else 0
        hidden = torch.zeros(1, model.hid_channels, *GRID_SIZE, device=device)

        model_x = []
        for k in range(model.input_length):
            s = state_history[-(k+1)]
            s = torch.from_numpy(s).float().unsqueeze(0).to(device)
            s = torch.cat([s, hidden], dim=1)

            model_x.append(s)

        action_map = None
        if model.actions > 1:
            action_map = torch.zeros(1, model.actions, *GRID_SIZE, device=device)
            action_map[0, action] = 1.0

        extra_map = torch.zeros(1, model.extra_channels, *GRID_SIZE, device=device)
        extra_map[0, 0, :, :] = 1.0

        with torch.no_grad():
            pred = model.step(model_x, action_map, extra_map, microsteps=MICROSTEPS)

        next_frame = pred[0, :model.vis_channels].argmax(dim=0).cpu().numpy()
        next_frame = np.eye(model.vis_channels)[next_frame].transpose(2, 0, 1)

        state_history.append(next_frame)
        state_history.pop(0)
    
    times = []
    for _ in range(num_runs):
        action = np.random.randint(0, model.actions) if model.actions > 1 else 0
        hidden = torch.zeros(1, model.hid_channels, *GRID_SIZE, device=device)

        model_x = []
        for k in range(model.input_length):
            s = state_history[-(k+1)]
            s = torch.from_numpy(s).float().unsqueeze(0).to(device)
            s = torch.cat([s, hidden], dim=1)

            model_x.append(s)

        action_map = None
        if model.actions > 1:
            action_map = torch.zeros(1, model.actions, *GRID_SIZE, device=device)
            action_map[0, action] = 1.0

        extra_map = torch.zeros(1, model.extra_channels, *GRID_SIZE, device=device)
        extra_map[0, 0, :, :] = 1.0
        
        torch.cuda.synchronize() if device == 'cuda' else None

        start = time.perf_counter()

        with torch.no_grad():
            pred = model.step(model_x, action_map, extra_map, microsteps=MICROSTEPS)

        torch.cuda.synchronize() if device == 'cuda' else None

        end = time.perf_counter()
        
        times.append(end - start)
        
        next_frame = pred[0, :model.vis_channels].argmax(dim=0).cpu().numpy()
        next_frame = np.eye(model.vis_channels)[next_frame].transpose(2, 0, 1)

        state_history.append(next_frame)
        state_history.pop(0)
    
    times_ms = np.array(times) * 1000
    
    print(f"FPS: {1000/times_ms.mean():.1f}")

if __name__ == "__main__":
    model.load("mario/renderer.pt")
    model.eval()

    print(f"Grid size: {GRID_SIZE[0]}x{GRID_SIZE[1]}")
    print(f"Microsteps: {MICROSTEPS}")
    print(f"Input length: {model.input_length}")

    print("\nNon-Compiled model:")
    test_inference_speed(model)

    print("\nCompiled model:")
    model = torch.compile(model, mode="reduce-overhead", backend="cudagraphs") # might not work for certain GPUs
    test_inference_speed(model)