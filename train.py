import torch, torch.nn as nn, torch.nn.functional as F
from collections import deque
from time import time
from tqdm import tqdm

from configs_vars import *
load_configuration()

if 'get_time_indices' not in globals():
    def get_time_indices(all_states, all_actions, file_indices):
        return [
            # all_actions or all_states - 1 is the same, since e.g: all_states has shape (602, 8, 8) and all_actions has shape (601,)
            torch.randint(len(all_actions[i]) - model.input_length - TRAIN_STEPS, (1,)).item() # with pooling, avoid t+2 being null
            for i in file_indices
        ]

if 'build_action_map' not in globals():
    def build_action_map(action_map, actions):
        action_map[torch.arange(BATCH_SIZE), actions] = 1.0 # set to 1.0 the taken action for each sample in the batch
        return action_map

if 'addnoise' not in globals():
    def addnoise(model_input): pass

gradient_clip_autodefine = False
if 'GRADIENT_CLIP' not in globals():
    GRADIENT_CLIP = 1.0
    gradient_clip_autodefine = True

if 'WEIGHT_LOSS' not in globals():
    if TRAIN_STEPS <= 4: # [1 to 4]
        # linear
        WEIGHT_LOSS = [i + 1 for i in range(TRAIN_STEPS)]

    elif TRAIN_STEPS <= 7: # [5 to 7]
        # squared
        from math import sqrt
        WEIGHT_LOSS = [sqrt(i + 1) for i in range(TRAIN_STEPS)]
        
    else: # [8 to inf]

        # normalization for stability (and avoid insane gradient clip values)
        raw_weights = [2 ** i for i in range(TRAIN_STEPS)]
        weight_sum = sum(raw_weights)

        # exponential
        WEIGHT_LOSS = [w / weight_sum * TRAIN_STEPS for w in raw_weights]

    # end

    if gradient_clip_autodefine:
        GRADIENT_CLIP = max(WEIGHT_LOSS) * 0.8

if POOL_LENGTH is not None:
    pool = deque(maxlen=POOL_LENGTH) # stores prediction, action_map, extra_map, target

def pad_to_same(tensors: list[torch.Tensor], pad_value=0.0) -> torch.Tensor:
    # ignore if all tensors have the same shape
    if all(t.shape == tensors[0].shape for t in tensors[1:]):
        return torch.stack(tensors)

    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)
    
    padded = []
    for t in tensors:
        h, w = t.shape[1], t.shape[2]
        padded.append(F.pad(t, (0, max_w - w, 0, max_h - h), value=pad_value))
    
    return torch.stack(padded)

scheduler: torch.optim.lr_scheduler._LRScheduler = globals().get('scheduler', None)

def main():
    # load data
    print("Loading data...")
    t = time()

    if LOAD_INSTANT and not LOAD_QUICK: # basically: (LOAD_INSTANT and not LOAD_QUICK) or (not LOAD_QUICK and LOAD_INSTANT):
        raise ValueError("LOAD_QUICK must be True if LOAD_INSTANT is True, LOAD_INSTANT must be False if LOAD_QUICK is False")
    
    all_states, all_actions, *extra_data = model.load_data(DATA_GLOB, limit=FILES_INCLUDE, load_quick=LOAD_QUICK, load_instant=LOAD_INSTANT, states='float', actions='long', **EXTRA_MAPS)
    all_extra = dict(zip(EXTRA_MAPS.keys(), extra_data))

    print(f"Done in {time() - t:.2f}ms")
    # print(f"Total samples: {len(all_states)}") # not correct

    # for graphs
    loss_history = []
    step_numbers = []

    # train loop
    for step in tqdm(range(STEPS), desc="Training"):
        if LOAD_QUICK:
            # random file indexes
            file_indices = torch.randint(len(all_states), (BATCH_SIZE,))
            
            # random frame indexes for each file
            time_indices = get_time_indices(all_states, all_actions, file_indices)
        else:
            # lazy loading:
            global_indices = torch.randint(len(all_actions) - model.input_length - TRAIN_STEPS, (BATCH_SIZE,))

        model_x = []
        for k in range(model.input_length):
            # padding so that files with different states sizes can be loaded while keeping BATCH_SIZE > 1
            if LOAD_QUICK:
                s = pad_to_same([
                    all_states[i][max(t - k, 0)] for i, t in zip(file_indices, time_indices)
                ])
            else:
                s = pad_to_same([
                    all_states[global_idx - k] for global_idx in global_indices
                ])
            
            if not LOAD_INSTANT:
                s = s.to(model.device)

            FILE_GRID_SIZE = (s.shape[2], s.shape[3]) # set after padding!

            #! NOTE: hidden channels are zeroed every single step!
            hidden_states = torch.zeros(BATCH_SIZE, model.hid_channels, *FILE_GRID_SIZE, device=model.device)

            s = torch.cat([s, hidden_states], dim=1) # append hidden channels to each
            model_x.append(s)

        # train step:
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=model.device)

        current_x = model_x

        for n in range(TRAIN_STEPS):
            if LOAD_QUICK:
                step_actions = torch.stack([
                    all_actions[i][min(t + n, len(all_actions[i]) - 1)] for i, t in zip(file_indices, time_indices)
                ])

                step_targets = pad_to_same([
                    all_states[i][min(t + n + 1, len(all_states[i]) - 1)] for i, t in zip(file_indices, time_indices)
                ])
            else: # lazy loading
                step_actions = torch.stack([
                    all_actions[global_idx + n] for global_idx in global_indices
                ])

                step_targets = pad_to_same([
                    all_states[global_idx + n + 1] for global_idx in global_indices
                ])
            
            if not LOAD_INSTANT:
                step_actions = step_actions.to(model.device)
                step_targets = step_targets.to(model.device)

            # action map for this step
            step_action_map = None
            if model.actions > 1:
                step_action_map = torch.zeros(BATCH_SIZE, model.actions, *FILE_GRID_SIZE, device=model.device)
                step_action_map = build_action_map(step_action_map, step_actions)

            # extra map for this step
            step_extra_map = None
            for extra in all_extra:
                if LOAD_QUICK:
                    thing = torch.stack([
                        all_extra[extra][i][min(t + n, len(all_extra[extra][i]) - 1)] for i, t in zip(file_indices, time_indices)
                    ])
                else:
                    thing = torch.stack([
                        all_extra[extra][global_idx + n] for global_idx in global_indices
                    ])
                
                if not LOAD_INSTANT:
                    thing = thing.to(model.device)

                # give info to all cells (concat with action)
                # NOTE! this assumes thing's shape is (B, C, H)
                thing_map = thing.unsqueeze(-1).expand(-1, -1, -1, FILE_GRID_SIZE[1])
                step_extra_map = thing_map if step_extra_map is None else torch.cat([step_extra_map, thing_map], dim=1)

            addnoise(current_x)
            
            model_pred = model.step(current_x, step_action_map, step_extra_map, microsteps=MICROSTEPS)
            total_loss += loss_calc(model_pred, step_targets) * WEIGHT_LOSS[n]

            pred_vis = model_pred[:, :model.vis_channels].detach()

            #! NOTE: hidden channels are zeroed every single step!
            hidden_states = torch.zeros(BATCH_SIZE, model.hid_channels, *FILE_GRID_SIZE, device=model.device)
            current_x = [torch.cat([pred_vis, hidden_states], dim=1)]

        # save to pool 40% of the time
        if POOL_LENGTH is not None and torch.rand(1).item() < 0.4:
            pass # unimplemented (i need to do more research first)
        
        total_loss.backward() # not normalized on purpose
        
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()

        # scheduler update
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(total_loss.item())
            else:
                scheduler.step()

        pool_loss: torch.Tensor = None
        # pool loss - separate step
        if POOL_LENGTH is not None and len(pool) > 10:
            pass # unimplemented (i need to do more research first)
        
        if (step+1) % LOG_SEGMENTS == 0:
            poolinfo = f" - Pool Loss: {pool_loss.item():.4f}" if pool_loss is not None else " - Pool Loss: None"
            lrinfo = f" - LR: {scheduler.get_last_lr()[0]:.3e}" if scheduler is not None else ""

            tqdm.write(f"Step {step+1}/{STEPS} - Loss: {total_loss.item():.4f}{"" if POOL_LENGTH is None else poolinfo}{lrinfo}")

            loss_history.append(total_loss.item())
            step_numbers.append(step+1)
    
    # (for loop end)
    
    # NOTE: loss_history is only used if LOSS_GRAPH is not None
    return step_numbers, loss_history

if __name__ == "__main__":
    from datetime import datetime

    LOG_SEGMENTS = STEPS // LOG_SEGMENTS

    t = time()
    try:
        print(f"Training for {STEPS} steps (With batch size {BATCH_SIZE})")

        # defined (maybe) in config
        if 'LOAD_MODEL' in globals():
            model.load(globals()['LOAD_MODEL'], optimizer=(optimizer if LOAD_OPTIMIZER else None))
        
        steps, losses, autoreg_losses = main()

        # save
        print(f"\nTraining completed in {time() - t:.2f}s.")

        model.save(FILE_NAME, optimizer=optimizer)

        print(f"Model saved as '{FILE_NAME}'\n")

        # plot graph
        if LOSS_GRAPH is not None:
            import matplotlib.pyplot as plt # ImportError!

            plt.figure(figsize=(10, 5))
            plt.plot(steps, losses, "b-", label="Training Loss")
            
            plt.xlabel("Step")
            plt.ylabel("Loss")

            plt.title("Loss Over Time")
            plt.grid(True, alpha=0.3)
            plt.legend()

            max_loss = max(losses)
            plt.ylim(0, min(max_loss, 1.5)) # show loss between 0 and clamped max loss (to avoid loss spikes to ruin the graph)

            plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
            plt.gca().yaxis.set_major_locator(plt.MaxNLocator(8)) # loss values that appear in the graph
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.4f}'))

            plt.savefig(LOSS_GRAPH, dpi=150, bbox_inches="tight")
            # plt.show()

            print(f"Loss graph saved as '{LOSS_GRAPH}'")

    except KeyboardInterrupt:
        print(f"\nTraining manually interrupted in {time() - t:.2f}s.")

        # save backup
        date = datetime.now().strftime("h%H-m%M")
        name = f"backup_{date}.pt"

        model.save(name, optimizer=optimizer)

        print(f"Backup saved as '{name}'\n")

    # raise all other exceptions