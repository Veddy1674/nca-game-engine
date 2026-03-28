import time
import numpy as np
import torch

from configs_vars import *
load_configuration()

def test_inference_speed(model: NACE, num_runs=500, warmup=50):
    device = model.device
    
    # Init state
    state = np.zeros((model.vis_channels, *GRID_SIZE), dtype=np.float32)
    state[-1] = 1.0
    state_history = [state.copy() for _ in range(model.input_length)]
    
    def prepare_inputs(action):
        hidden = torch.zeros(1, model.hid_channels, *GRID_SIZE, device=device)
        model_x = [torch.cat([torch.from_numpy(state_history[-(k+1)]).float().unsqueeze(0).to(device), hidden], dim=1) 
                   for k in range(model.input_length)]
        
        # actions
        action_map = None
        if model.actions > 1:
            action_map = torch.zeros(1, model.actions, *GRID_SIZE, device=device)
            action_map[0, action] = 1.0
        
        # extra channels
        extra_map = None
        if model.extra_channels > 0:
            extra_map = torch.zeros(1, model.extra_channels, *GRID_SIZE, device=device)
            extra_map[0, 0] = 1.0
        
        return model_x, action_map, extra_map
    
    def update_state(pred):
        next_frame = pred[0, :model.vis_channels].argmax(dim=0).cpu().numpy()
        next_frame = np.eye(model.vis_channels)[next_frame].transpose(2, 0, 1)
        state_history.append(next_frame)
        state_history.pop(0)
        return next_frame
    
    # Warmup
    for _ in range(warmup):
        action = np.random.randint(0, model.actions or 1)
        with torch.no_grad():
            pred = model.step(*prepare_inputs(action), microsteps=MICROSTEPS)
        update_state(pred)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        action = np.random.randint(0, model.actions or 1)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            pred = model.step(*prepare_inputs(action), microsteps=MICROSTEPS)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        
        update_state(pred)
    
    times_ms = np.array(times) * 1000
    print(f"FPS: {1000/times_ms.mean():.1f}")

if __name__ == "__main__":
    model.load(MODEL_PATH)
    model.eval()
    
    print(f"Input Dimension: {model.input_dim}\nMicrosteps: {MICROSTEPS}\nInput length: {model.input_length}\nDilation: {model.dilations}")
    
    print("\nNon-Compiled:")
    test_inference_speed(model)
    
    print("\nCompiled:")
    model = torch.compile(model, mode="reduce-overhead", backend="cudagraphs")
    test_inference_speed(model)