---

![](https://github.com/ProofofLearning/images/watermark-trajectory.png)
# Privacy-Preserving Proof-of-Learning via Watermark Trajectory (WT-PoL)

## Introduction

Welcome to the official repository of the "Privacy-Preserving Proof-of-Learning via Watermark Trajectory" project. Our work introduces an innovative Proof-of-Learning (PoL) method, WT-PoL, which is hinged on a unique watermarking mechanism we call "watermark trajectory". This method is specifically designed for scenarios where ensuring the integrity and authenticity of the learning process is critical.

### Key Features

- **Watermark Trajectory**: A novel approach that alternates between embedding and removing watermarks during training.
- **Abnormal Training Detection**: Tracks abnormal iterations in training and adjusts the watermark trajectory accordingly.
- **High Accuracy**: Demonstrates up to 100% accuracy in detecting abnormal training iterations, even with as low as 0.05 proportion of such iterations.
- **Minimal False Accusations**: Designed to minimize false positives in abnormal training detection.

## How WT-PoL Works

WT-PoL operates by creating multiple sub-trajectories throughout the training process. Each sub-trajectory monitors different stages of training, specifically targeting abnormal training iterations. By making the watermark sensitive to these abnormalities, any deviation from the normal training process is quickly detected, ensuring the integrity of the learning process.

## System Requirements

**Python Version**: Python 3.8
**PyTorch Version**: PyTorch 1.8.1

## Dependencies

The training process for the watermark generator (**'1train_wm_generator.py'**) and the anti-watermark generator (**'2train_anti_wm_generator.py'**) depends on an external repository for imperceptible backdoor attacks. You can clone this repository using the following command:

```bash
git clone https://github.com/Ekko-zn/IJCAI2022-Backdoor
```

## Running the Code

To run the WT-PoL system, follow these steps in your terminal or command prompt:

1. **Generate the Initial Watermark**: 
   ```bash
   python 1train_wm_generator.py
   ```

2. **Train the Anti-Watermark Generator**: 
   ```bash
   python 2train_anti_wm_generator.py
   ```

3. **Select Pivot Samples**: 
   ```bash
   python 3pivot_sample_selection.py
   ```

4. **Generate the Proof**: 
   ```bash
   python 4run_proof_generation.py
   ```

5. **Run the Verification Process**: 
   ```bash
   python 5run_verification.py
   ```

## Contributing

We welcome contributions to this project! If you have suggestions or improvements, please fork the repository and submit a pull request.

