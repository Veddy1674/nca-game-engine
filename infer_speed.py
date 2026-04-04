import time
import numpy as np
import torch

from configs_vars import *
load_configuration()

def test_inference_speed(model: NACE, num_runs=500, warmup=50):
    device = model.device
    
    state = torch.zeros((1, model.vis_channels, *GRID_SIZE), dtype=torch.float32, device=device)
    state[:, -1] = 1.0
    
    hidden = torch.zeros(1, model.hid_channels, *GRID_SIZE, device=device)
    state_history = [state.clone() for _ in range(model.input_length)]
    
    action_map = torch.zeros(1, model.actions, *GRID_SIZE, device=device) if model.actions > 1 else None
    extra_map = torch.zeros(1, model.extra_channels, *GRID_SIZE, device=device) if model.extra_channels > 0 else None
    
    def prepare_inputs(action):
        model_x = [torch.cat([state_history[-(k+1)], hidden], dim=1) for k in range(model.input_length)]
        
        if action_map is not None:
            action_map.zero_()
            action_map[0, action] = 1.0
        
        if extra_map is not None:
            extra_map.zero_()
            extra_map[0, 0] = 1.0
        
        return model_x, action_map, extra_map
    
    for _ in range(warmup):
        action = np.random.randint(0, model.actions or 1)
        with torch.no_grad():
            pred = model.step(*prepare_inputs(action), microsteps=MICROSTEPS)
        
        next_frame = torch.nn.functional.one_hot(pred[0, :model.vis_channels].argmax(dim=0), model.vis_channels).permute(2, 0, 1).unsqueeze(0).float()
        state_history.append(next_frame)
        state_history.pop(0)
    
    times = []
    if device == 'cuda':
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
    
    for _ in range(num_runs):
        action = np.random.randint(0, model.actions or 1)
        
        if device == 'cuda':
            starter.record()
        else:
            start = time.perf_counter()
        
        with torch.no_grad():
            pred = model.step(*prepare_inputs(action), microsteps=MICROSTEPS)
        
        if device == 'cuda':
            ender.record()
            torch.cuda.synchronize()

            times.append(starter.elapsed_time(ender))
        else:
            times.append((time.perf_counter() - start) * 1000)
        
        next_frame = torch.nn.functional.one_hot(pred[0, :model.vis_channels].argmax(dim=0), model.vis_channels).permute(2, 0, 1).unsqueeze(0).float()
        state_history.append(next_frame)
        state_history.pop(0)
    
    times_ms = np.array(times)
    print(f"FPS: {1000/times_ms.mean():.1f}")

if __name__ == "__main__":
    model.load(MODEL_PATH)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Input dimension: {model.input_dim}")
    print(f"Kernel size: {model.kernel_size}")
    print(f"Total parameters: {total_params:,}")
    print(f"Microsteps: {MICROSTEPS}")
    print(f"Input length: {model.input_length}")
    print(f"Dilation: {model.dilations}")
    
    print("\nNon-Compiled:")
    test_inference_speed(model)
    
    print("\nCompiled:")
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    test_inference_speed(model)