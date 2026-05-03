![Status](https://img.shields.io/badge/STATUS-RESEARCHING-lightblue?style=plastic)

<img width="286" height="88" alt="NACE_logo" src="https://github.com/user-attachments/assets/04fc1d2c-987a-4710-8629-f08161dcb108" />

NACE (Neural Adaptive Cellular Engine) is an AI architecture designed to simulate interactive environments through emergent behaviors: no hardcoded physics or game logic, just a neural network predicting the next frame from the current state and player input.

[Architecture](#architecture)ㆍ[Project Structure](#project-structure)ㆍ[Getting Started](#getting-started)

## Preview:

<sub>Click on the previews to view the video</sub>

### Falling Sand  
<a href="https://youtube.com/shorts/ne5x8hp67sg?feature=share">
  <img src="https://github.com/user-attachments/assets/7730db42-b4a9-473e-bcdf-5a28f7d79447" width="400" height="400">
</a>
<p>
  <em>A minimal sand simulation environment where a controllable emitter generates falling sand particles. (The gradient was applied as a post-processing effect, not learnt by the model)</em>

  <details>
  <summary>More Info</summary>

  This is one of the best examples of what this architecture is good at learning: local simulations, because NCAs are used to simulate water, sand, fire physics, and this architecture is heavily inspired by NCAs, which is why it learns in a relatively short amount of time:

  <img width="1308" height="701" alt="image" src="https://github.com/user-attachments/assets/c24e96ee-76fd-470c-b93b-7e68fe44de49" />
  <em>The loss curve shows fluctuations mostly because noise was applied during training for stability purposes</em>

  The model was trained on about 18.000 total samples for this preview, which is tiny and takes less than a minute to train on modern GPUs.

  At the end, the model runs at:  
  <b>~1489.9 FPS</b> on a <b>RTX 5060</b> GPU  
  <b>~208.0 FPS</b> on a <b>Ryzen 7 5700X</b> CPU  
  (Both tests on a non-compiled and unoptimized model and a single batch, meaning it could run even faster)  

  </details>
</p>

---

### Super Mario Bros. Level Generator
<a href="https://www.youtube.com/shorts/_pifXhjWY2g">
  <img src="https://github.com/user-attachments/assets/6eaadec9-a884-4450-bcb2-eefdc51be50f" width="400" height="400">
</a>
<p>
  <em>The classic Super Mario Bros. from the NES console, except it's just a real-time level generator</em>

  <details>
  <summary>More Info</summary>

  The dataset it was trained on contains all the 13 overworld maps of the original game: underground, castle and water levels were purposely excluded, so that the sky is one single color and the prediction isn't "half sky half underground half water" when temperature is applied.

  The model only has to predict a 15x15 grid, each pixel representing a 16x16 sprite: each sprite appearing across the 13 maps is assigned a unique color, effectively turning each sprite into a separate class/object for the model.  
  This way, during inference the window is 240x240, and interpolation between frames is used to avoid snappy movements, which also improves performance as fewer model forwards are done per second.

  <img width="1308" height="701" alt="image" src="https://github.com/user-attachments/assets/800fbe95-6583-4b6c-b9c9-d9548faa5906" />
  <em>The loss curve shows fluctuations mostly because noise was applied during training for stability purposes</em>

  The model was trained on about 960.000 total samples for this preview, quite a small amount considering how well it learns to reproduce most of the levels.

  This environment shows the case where an additional parameter is used: the number of the level, so that the AI knows what level style to get inspired by, for example if it's a snowy level, it will use snowy assets instead of green. (This is not necessary, although without it, with a higher temperature it mixes up different styles)

  In this preview, the model has learnt to reproduce the levels almost perfectly; by increasing the temperature, the model becomes a level generator, which draws tiles in uncommon patterns while remaining mostly coherent with the game's design.

  At the end, the model runs at:  
  <b>~426.2 FPS</b> on a <b>RTX 5060</b> GPU  
  <b>~171.6 FPS</b> on a <b>Ryzen 7 5700X</b> CPU  
  (Both tests on a non-compiled and unoptimized model and a single batch, meaning it could run even faster)  

  </details>
</p>

---

## Architecture

Everything is based on the equation $f(s, a) = \Delta s$ <b>and</b> $s + \Delta s = s'$.  
A neural network processes the current state ($s$) and the player action ($a$) to predict a residual delta ($\Delta s$).  
This delta is then added to the current state to compute the next frame ($s'$) in an autoregressive loop, where each prediction becomes the input for the next one.  
<i>Note that this is the "default configuration", you could try different approaches such as $f(s, a) = s'$ directly, add more inputs, or even not use it as a image predictor model at all</i>

Each cell/pixel in the grid evolves based on its <b>local neighborhood</b> to learn <b>rules</b>, for instance, a cell could 'think': <i>If all the cells around me are sky, I will become sky too in the next state</i>, or <i>If I am the bottom part of the player, and the global action was up, I will become background in the next state</i>.

Cells can "read" other cells' classes/colors and hidden channels, and decide what to become.  
Unlike [Growing NCA](https://distill.pub/2020/growing-ca/), which focuses on self-organization, <b>NACE</b> is <b>built for interactive environments</b>, resulting in a true <b>engineless simulation</b> where physics, logic and graphics are all learnt by the <i>pixels</i> themselves.

Another difference is that, <i>by default</i>, this architecture has no persistent memory: hidden channels only persist during the microsteps, meaning every frame the cells forget everything; this promotes stability and precision on single-state generation.  
However, persistent hidden channels across frames are also supported, which can be useful when the model benefits from longer temporal context.

Basically, hidden channels in NACE act as a local communication layer: during the microsteps, cells 'pass messages' to their neighbors to build spatial awareness.

A simplified analogy of how this works:
<i>
If a cell in the center has to know the distance to the border beyond its perception, the cells in between will 'pass the information', each writing down in their hidden channels something like: "Hey, I'm a border!", "On my left there's a cell who said they're a border!", "On my left there's a cell who said on their left there's a cell who said they're a border!" ... until they get to the central cell who requires the information. - Of course in reality it's just unintelligible floating-point numbers that only the neural network understands.
</i>

Unlike heavier world models approaches (e.g: JEPA, Dreamer V3), this architecture is lightweight, fast, and meant to be run easily on consumer GPUs rather than expensive TPUs.  
It's very limited in comparison, but excels in learning local rules in simple games and simulations.  
<em>(A notable example of diffusion-based world model is [Oasis by Decart](https://oasis.decart.ai/introduction))</em>

---

## Project Structure

| File | Description |
|------|-------------|
| `NACE.py` | The model's architecture (defines class 'NACE' and 'Dataset') |
| `train.py` | To train and fine-tune models |
| `visualizer_cv2.py` | Inference via opencv-python, to try out the model |
| `visualizer_pygame.py` | Same as above, but via pygame - smoother graphics and continuous actions |
| `infer_speed.py` | To benchmark inference speed and view information about the model |
| `configs_vars.py` | Contains info about what each parameter does in the configurations (see [Getting Started](#getting-started)) |
---

## Getting Started

Prerequisites:
- Python 3.10+ (3.12.3 recommended).
- PyTorch 2.0+, which you can find [here](https://pytorch.org/get-started/locally/).

1. Clone the repository via `git clone https://github.com/Veddy1674/nca-game-engine.git`.
2. Install the dependencies with `pip install -r requirements.txt`.
3. If you are using VSCode, I highly recommend setting up <b>tasks</b> to easily switch between training configurations:

'train.py' and all the other scripts for inference, testing, etc, all require a path to a configuration file as the first argument:  
So you could create a task in VSCode to run scripts and set their first argument to the file you're currently viewing, this allows you to simply open a config.py file and hit the shortcut for `Tasks: Run Task`:  
```json
{
    "label": "NACE: Train with current config",
    "type": "shell",
    "command": "python", // Your python path/virtual env
    "args": [
        "${workspaceFolder}/train.py",
        "${relativeFile}"
    ],
    "group": "build",
    "presentation": {
        "reveal": "always",
        "panel": "new"
    }
}
```

Alternatively, execute scripts manually, e.g: `python train.py example/config.py`.

---

The `example/` directory contains a minimal environment and a pre-trained model to try out.

Before anything else, run [testview.py](example/dev/testview.py) to see (and play) the environment the model will learn to simulate - the actual game/simulation.

You can create a dataset (.npz files) by running [env.py](example/dev/env.py).

Then, have a look at the [configuration file](example/config.py) to get an idea of how each parameter affects the model training and inference. (More info in [configs_vars.py](configs_vars.py)).

Train the model with `python train.py example/config.py` (or run the task).

Finally, try out the model with `python visualizer_cv2.py example/config.py` (or run the task).  
If 'LOSS_GRAPH' is not None in the configuration file, a [loss graph](example/loss_graph.png) will be saved in the specified path.

Feel free to tweak the configuration: loss function, scheduler, weights, microsteps, or override `post_processing()` and other functions in [visualizer_cv2.py](visualizer_cv2.py) for custom inference effects.