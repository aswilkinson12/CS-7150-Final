# CS-7150-Final

## LSTM Model

📦 Model Architecture:
  - Input: (x, y, vx, vy, ax, ay) = 6 features
  - Encoder: Bidirectional LSTM (128 hidden, 2 layers)
  - Decoder: 3-layer MLP with dropout
  - Output: Residual displacement prediction
  - Parameters: 585,740

📊 Training Configuration:
  - Epochs: 200
  - Final Train Loss: 0.4161
  - Final Val Loss: 1.3636

🎯 Performance Metrics:
  - ADE (Average Displacement Error): 2.7315 m
  - FDE (Final Displacement Error): 4.8633 m
  - ADE Std: 3.2118 m
  - 90th Percentile ADE: 8.3908 m

✅ Key Components:
  1. Velocity + Acceleration features
  2. Bidirectional LSTM encoding
  3. Residual (displacement) prediction
  4. Deeper MLP decoder
  5. Better regularization (Dropout, Weight Decay)
  6. Smoother loss function (SmoothL1)

🔔 How to run:
  - Just run the notebook
  - Pay attention to the dataset path

⚠️ Very important bug fix from the original LSTM model:
  - Problem 1: Incorrect Coordinate Normalization. Samples from later in the scene have coordinates reaching 50-100m. Model must predict absolute positions with large numerical range. This makes learning extremely difficult
  - Solution 1: Normalize EACH sample individually, using last past point as origin

  - Problem 2: Improper Train/Val Split. Sequential split: scenes 1-8 for training, scenes 9-10 for validation. Train and val sets come from different scenes with different distributions
  - Solution 2: Random shuffle before split ensures consistent distribution