![Status](https://img.shields.io/badge/STATUS-RESEARCHING-lightblue?style=plastic)

<img width="286" height="88" alt="NACE_logo" src="https://github.com/user-attachments/assets/04fc1d2c-987a-4710-8629-f08161dcb108" />

NACE (Neural Adaptive Cellular Engine) is an AI architecture designed to simulate game-like environments through emergent behaviors: no hardcoded physics or game logic, just a neural network learning to predict the next frame based on the current state and player actions.

## Preview:

TODO

# Architecture

Everything is based on this equation: <b>s + a = s'</b> (eventually with additional inputs), the neural network is simply fed

Each cell in the grid evolves based on its local neighborhood, hidden internal state, and global context, producing pixel-perfect simulations at 60fps on consumer GPUs.

---

## Demos

<figure style="text-align: center; width: 240px; margin: 0;">
  <figcaption style="font-weight: bold; margin-bottom: 5px;">Falling Sand</figcaption>
  <img src="https://github.com/user-attachments/assets/41a507d2-1813-4bea-b8e9-67b5e9ebb29e" width="240" style="display: block;">
</figure>


Simulates sand particles, floor collisions and spawn interactions. No hardcoded physics — just a NCA predicting the next frame.

<details>
<summary>Details</summary>

- **Grid Size:** Trained on 48x48, predicting on 96x96
- **Channels:** 4 Visible (floor, rectangle, sand, background), 16 Hidden
- **Input Length:** 2 Frames (Visible channels only)
- **Microsteps:** 12
- **Actions:** 4 (NO-OP, right, left, spawn sand)
- **Loss Function:** CrossEntropyLoss with weights
- **Train steps:** 3.000 steps (fine-tuned loss weights each 1.000)

</details>

---

## Architecture
...

## Project Structure
...

## Getting Started
...