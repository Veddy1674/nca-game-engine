![Status](https://img.shields.io/badge/STATUS-RESEARCHING-lightblue?style=plastic)

<img width="286" height="88" alt="NACE_logo" src="https://github.com/user-attachments/assets/04fc1d2c-987a-4710-8629-f08161dcb108" />

NACE (Neural Adaptive Cellular Engine) is an AI architecture designed to simulate game-like environments through emergent behaviors: no hardcoded physics or game logic, just a neural network learning to predict the next frame based on the current state and player actions.

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

  This is one of the best examples of what this architecture is good at learning: local simulations, originally, NCAs are used to simulate water, sand, fire physics, and this architecture is heavily inspired by NCAs, which is why it learns in a relatively short amount of time:

  <img width="1308" height="701" alt="image" src="https://github.com/user-attachments/assets/c24e96ee-76fd-470c-b93b-7e68fe44de49" />
  <em>The loss curve shows fluctuations mostly because a noise was applied during training for stability purposes</em>

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
  <em>The loss curve shows fluctuations mostly because a noise was applied during training for stability purposes</em>

  The model was trained on about 960.000 total samples for this preview, a quite small amount considering how well it learns to reproduce most of the levels.

  This environment shows the case where an additional parameter is used: the number of the level, so that the AI knows what level style to get inspired by, for example if it's a snowy level, it will use snowy assets instead of green. (This is not necessary, although without it, with a higher temperature it mixes up different styles)

  In this preview, the model has learnt to reproduce the levels almost perfectly, by increasing the temperature instead, it becomes a level generator, which draws tiles in uncommon patterns, while still being coherent with the game's style <i>(not always)</i>.

  At the end, the model runs at:  
  <b>~426.2 FPS</b> on a <b>RTX 5060</b> GPU  
  <b>~171.6 FPS</b> on a <b>Ryzen 7 5700X</b> CPU  
  (Both tests on a non-compiled and unoptimized model and a single batch, meaning it could run even faster)  

  </details>
</p>

---

## Architecture

Everything is based on the equation <b>s + a = s'</b> (eventually with additional inputs), the neural network is simply fed the current state/frame/image and the player action, and predicts the next state/frame/image, autoregressively.

Each cell/pixel in the grid evolves based on its local neighborhood and learns <b>rules</b>, for example, a cell could 'think': <i>If all the cells around me are sky, I will become sky too in the next state</i>, or <i>If I am the bottom part of the player, and the global action was up, I will become background in the next state</i>.

Each cell does not change abruptly, but rather, can "move" around, "read" other cells' class/color and hidden channels, and decide what to become.  
Unlike [Growing NCA](https://distill.pub/2020/growing-ca/), this architecture focuses on external user input, creating a sort of game, except there is no engine and every single frame is AI generated.

Another difference is that this architecture has no <b>memory</b>, in growing NCA, hidden channels are given to each cell as persistent memory, in this architecture hidden channels instead only persist during the microsteps, meaning every frame the cells forget everything; this promotes stability and precision on single-state generation, but might be unstable on longer sequences, to overcome this, 'input_length' makes it so each cell can also see what information was it there in previous frames, the formula becomes something like <b>s + s' + a = s''</b>.

Unlike diffusion models (e.g: JEPA, Dreamer V3), this approach focuses on <b>real-time, pixel-perfect frame predictions, lightweight and fast enough to run on consumer GPUs.</b>
<em>(A good example of diffusion model is [Oasis by Decart](https://oasis.decart.ai/introduction))</em>

---

## Project Structure
TODO

## Getting Started
TODO