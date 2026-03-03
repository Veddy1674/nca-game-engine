import torch, torch.nn as nn
from time import time
from tqdm import tqdm

from colorchange.configRGB import *

def main():
    # load data
    print("Loading data...")
    t = time()

    all_states, all_actions = model.load_data(DATA_GLOB)

    print(f"Done in {time() - t:.2f}ms")

    # train loop
    for step in tqdm(range(STEPS)):

        # random file indexes
        file_indices = torch.randint(len(all_states), (BATCH_SIZE,))
        
        # random frame indexes for each file
        time_indices = [
            # all_actions or all_states - 1 is the same, since e.g: all_states has shape (602, 8, 8) and all_actions has shape (601,)
            torch.randint(len(all_actions[i]) - model.input_length, (1,)).item()
            for i in file_indices
        ]

        # get visible channels for the current states/frames
        states = torch.stack([
            all_states[i][t] for i, t in zip(file_indices, time_indices)
        ]).to(model.device)

        # a
        actions = torch.stack([
            all_actions[i][t] for i, t in zip(file_indices, time_indices)
        ]).to(model.device)
        
        # s'
        targets = torch.stack([
            all_states[i][t + 1] for i, t in zip(file_indices, time_indices)
        ]).to(model.device)

        # add zeroed hidden channels to the input (if input_length == 1)
        # hidden_states = torch.zeros(
        #     BATCH_SIZE, model.hid_channels, *GRID_SIZE, 
        #     device=model.device
        # )

        hidden_states = torch.zeros(BATCH_SIZE, model.hid_channels, *GRID_SIZE, device=model.device)

        model_x = []
        for k in range(model.input_length):
            # add to input the past k states (if input_length >= 1)
            s = torch.stack([
                all_states[i][max(t - k, 0)] for i, t in zip(file_indices, time_indices)
            ]).to(model.device)

            s = torch.cat([s, hidden_states], dim=1) # append hidden channels to each
            model_x.append(s)

        # (BATCH_SIZE, vis_channels + hid_channels, H, W) e.g: (128, 16, 8, 8)
        # model_x = torch.cat([states, hidden_states], dim=1) # ONLY FOR input_length == 1

        # (BATCH_SIZE, model.actions, H, W) e.g: (128, 4, 8, 8) = [0, 0, 0, 0]

        if model.actions > 1:
            action_map = torch.zeros(BATCH_SIZE, model.actions, *GRID_SIZE, device=model.device)
            action_map[torch.arange(BATCH_SIZE), actions] = 1.0 # set to 1.0 the taken action for each sample in the batch
        else:
            action_map = None

        # train step:
        optimizer.zero_grad()

        model_pred = model.step(model_x, action_map, microsteps=MICROSTEPS)

        # get predicted visible channels
        pred_visible = model_pred[:, :model.vis_channels]
        
        loss: torch.Tensor = loss_calc(pred_visible, states, targets) # customizable for each config

        loss.backward()
        
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        
        if (step+1) % LOG_SEGMENTS == 0:
            tqdm.write(f"Step {step+1}/{STEPS}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    from datetime import datetime

    LOG_SEGMENTS = STEPS // LOG_SEGMENTS

    t = time()
    try:
        print(f"Training for {STEPS} steps (With batch size {BATCH_SIZE})")

        # defined (maybe) in config
        if 'LOAD_MODEL' in globals():
            model.load(globals()['LOAD_MODEL'], optimizer=optimizer)
        
        main()

        # save
        print(f"\nTraining completed in {time() - t:.2f}s.")

        model.save(FILE_NAME, optimizer=optimizer)

        print(f"Model saved as '{FILE_NAME}'\n")

    except KeyboardInterrupt:
        print(f"\nTraining manually interrupted in {time() - t:.2f}s.")

        # save backup
        date = datetime.now().strftime("h%H-m%M")
        name = f"backup_{date}.pt"

        model.save(name, optimizer=optimizer)

        print(f"Backup saved as '{name}'\n")

    # raise all other exceptions