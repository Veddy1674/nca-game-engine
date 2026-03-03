Experimenting with Neural Cellular Automata to work as a Neural Game Engine: realtime frame generation, 60fps, lightweight and pixel-perfect precision

Here a list of environments the NCA was trained on:
- <b>Name: </b>Falling Sand 
<img src="https://github.com/user-attachments/assets/41a507d2-1813-4bea-b8e9-67b5e9ebb29e" width="300" autoplay>

- <b>Grid Size: </b>Trained on 48x48, predicting on 96x96  
- <b>Channels: </b>4 Visible (floor, rectangle, sand, background), 16 Hidden  
- <b>Input Length: </b>2 Frames (Visible channels only)  
- <b>Microsteps: </b>12  
- <b>Actions: </b>4 (NO-OP, right, left, spawn sand)  
- <b>Loss Function: </b>CrossEntropyLoss with weights  
- <b>Train steps: </b>Only 3.000 steps total (fine-tuned loss weights each 1.000)  
