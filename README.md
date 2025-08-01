# NGF-Inspired Pruning for Efficient Neural Networks (Supervised & RL)

This project implements and evaluates a neural network pruning technique inspired by the biological phenomenon of neuronal apoptosis mediated by Neural Growth Factor (NGF). So far, I've tested basic supervised learning tasks (image classification) and reinforcement learning tasks (continuous control).

## Abstract

The main idea is to emulate the biological process where neurons compete for limited resources (like NGF), and those failing to achieve sufficient activation during critical developmental periods are eliminated. This project applies this concept to artificial neural networks by monitoring and selectively pruning neurons exhibiting persistently low activation during a predetermined "critical period" in training. We also implement standard magnitude-based pruning for comparison. The effectiveness of these pruning methods is evaluated across different simple network architectures (MLPs, CNNs, Policy Networks) and tasks (MNIST, Fashion-MNIST, CartPole-v1), comparing performance against baseline (unpruned) models.

## Introduction

Traditional pruning methods often focus on weight magnitudes post-training or apply pruning uniformly. This project explores an alternative, biologically-inspired approach (NGF) where pruning is based on *neuron activation* during a specific phase of training. This allows the network to adapt dynamically to structural changes. We compare this NGF-inspired method (both single-shot and scheduled) against standard magnitude pruning and unpruned baselines in both supervised and simple reinforcement learning settings.

## Methods

### Pruning Techniques
1.  **NGF-Inspired Pruning (Single-Shot):**
    *   Neurons' average activation magnitudes are monitored during a defined "critical period" (epochs for SL, episodes for RL).
    *   At the end of the period, neurons whose activation falls below a threshold (determined by a `keep_fraction`) are pruned. Pruning involves masking outputs and zeroing corresponding weights/biases in custom `PrunableLinear` and `PrunableConv2d` layers.
2.  **NGF-Inspired Pruning (Scheduled):**
    *   Similar to single-shot, but pruning is applied iteratively over a defined schedule (start epoch/episode, end epoch/episode, frequency).
    *   The `keep_fraction` typically decreases over the schedule (e.g., polynomial decay), allowing for gradual pruning.
3.  **Magnitude Pruning (Baseline):**
    *   Standard global unstructured pruning based on weight magnitude (L1 norm).
    *   Applied once at a specified epoch (SL) or episode (RL). Weights below a threshold (determined by a `prune_fraction`) are zeroed out and the pruning is made permanent.

### Training Paradigms
1.  **Supervised Learning:** Standard training loop with epochs, batch processing, loss calculation (CrossEntropy), and backpropagation. Includes early stopping based on validation accuracy. Uses `src/trainer.py`.
2.  **Reinforcement Learning (REINFORCE):**
    *   Uses the REINFORCE (Monte Carlo Policy Gradient) algorithm.
    *   Trains an agent on a Gymnasium environment (like `CartPole-v1`).
    *   Collects trajectories (state, action, reward) for each episode.
    *   Calculates discounted returns.
    *   Updates the policy network based on sampled actions and calculated returns. Uses `src/rl_trainer.py`.
    *   Includes periodic saving of episode videos.

## Experiments

So farm the basic experiments evaluate pruning methods on:
*   **Tasks:**
    *   Supervised: Image Classification (MNIST, Fashion-MNIST)
    *   Reinforcement Learning: Classic Control (CartPole-v1)
*   **Models:**
    *   Supervised: Multi-layer Perceptrons (`SimpleMLP`), LeNet-5 (`LeNet5`)
    *   RL: Policy Network (`PolicyMLP`)
*   **Metrics:**
    *   Supervised: Accuracy (test/best), loss, pruning rate, parameter count reduction, training time.
    *   RL: Episode rewards (raw/average), episode lengths, pruning rate, training time.
*   **Comparisons:** Performance is compared across pruning methods ('none', 'ngf', 'magnitude', 'ngf_scheduled') and against unpruned baselines within each task.
*   **Hyperparameter Analysis:** The impact of varying pruning timing (start/duration/apply epoch/episode), thresholds (keep/prune fraction), and scheduling parameters is investigated.

## Project Structure

```
.
├── main.py                 # Main script to run experiments (supports all pruning methods)
├── requirements.txt        # Python package dependencies
├── run_experiments.sh      # Example script for batch runs (supervised)
├── plot_results.py         # Script to generate comparison plots (supervised)
├── README.md               # This file
├── results/                # Directory for saving experiment outputs
│   ├── supervised/         # Results for supervised tasks
│   │   ├── <dataset>/
│   │   │   ├── <model>/
│   │   │   │   ├── <run_name>/ # Individual run results
│   │   │   │   │   ├── config.json
│   │   │   │   │   ├── results.json
│   │   │   │   │   ├── training_plots.png
│   │   │   │   │   └── final_model.pth
│   │   └── ...
│   ├── rl/                 # Results for RL tasks
│   │   ├── <env_name>/
│   │   │   ├── <model>/    # e.g., policy_mlp
│   │   │   │   ├── <run_name>/ # Individual run results
│   │   │   │   │   ├── config.json
│   │   │   │   │   ├── results.json
│   │   │   │   │   ├── rl_training_plots.png
│   │   │   │   │   ├── final_model.pth
│   │   │   │   │   └── videos/ # Saved episode videos
│   │   └── ...
│   └── comparison_plots/   # Aggregated comparison plots (from plot_results.py)
└── src/                    # Source code
    ├── constants.py        # Project constants and configuration values
    ├── data_loader.py      # Data loading for supervised tasks
    ├── models.py           # Model definitions (MLP, LeNet5, PolicyMLP)
    ├── pruning.py          # Pruning logic (Prunable layers, hooks, calcs)
    ├── trainer.py          # Trainer class for supervised learning
    ├── rl_env.py           # RL environment setup (CartPole)
    └── rl_trainer.py       # Trainer class for reinforcement learning
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd NGF-PRUNE
    ```
2.  **Install dependencies:**
    *   **System Dependencies (Linux):** You might need development libraries for `pygame` (a dependency of Gymnasium classic control).
        *   *Debian/Ubuntu:* `sudo apt-get update && sudo apt-get install -y libfreetype6-dev libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libportmidi-dev python3-dev`
        *   *Fedora:* `sudo dnf install -y freetype-devel SDL2-devel SDL2_image-devel SDL2_mixer-devel SDL2_ttf-devel portmidi-devel python3-devel`
    *   **Python Packages:**
        ```bash
        pip install -r requirements.txt
        ```

## Usage

Use `main.py` to run experiments. Select the task (`--task`), model, dataset/environment, pruning method, and associated parameters.

**Key Arguments:**

*   `--task`: `supervised` or `rl` (default: `supervised`)
*   `--pruning-method`: `none`, `ngf`, `magnitude`, `ngf_scheduled` (default: `none`)
*   `--seed`: Random seed (default: 42)
*   `--gpu`: GPU ID to use (default: None for CPU)
*   `--lr`: Learning rate (default: 0.001)
*   `--results-dir`: Base directory for results (default: `results`)
*   `--pruning-threshold`: Meaning depends on method (NGF: keep fraction, Magnitude: prune fraction)

**Supervised Task Arguments (`--task supervised`):**

*   `--dataset`: `mnist`, `fashion_mnist`
*   `--model`: `mlp`, `lenet`
*   `--batch-size`: (default: 128)
*   `--epochs`: Max epochs (default: 50)
*   `--early-stopping-patience`: (default: 4)
*   `--critical-start-epoch`, `--critical-duration`: For single-shot NGF
*   `--pruning-apply-epoch`: For magnitude pruning
*   *(Scheduled NGF args below also apply, using epochs)*

**RL Task Arguments (`--task rl`):**

*   `--env-name`: Gymnasium env ID (default: `CartPole-v1`)
*   `--episodes`: Training episodes (default: 1000)
*   `--gamma`: Discount factor (default: 0.99)
*   `--video-every-n-episodes`: Save video frequency (0=disable, default: 100)
*   `--critical-start-episode`, `--critical-duration-episodes`: For single-shot NGF
*   `--pruning-apply-episode`: For magnitude pruning
*   *(Scheduled NGF args below also apply, using episodes)*

**Scheduled NGF Arguments (Applies to both tasks, unit is epochs/episodes):**

*   `--pruning-start-epoch`: Start epoch/episode for schedule
*   `--pruning-end-epoch`: End epoch/episode for schedule (exclusive)
*   `--pruning-frequency`: Steps between pruning applications
*   `--initial-keep-fraction`: Starting keep fraction
*   `--final-keep-fraction`: Target keep fraction
*   `--schedule-power`: Polynomial decay exponent (default: 3.0)

**Examples:**

```bash
# === Supervised ===
# Train LeNet5 on FashionMNIST, no pruning
python main.py --task supervised --dataset fashion_mnist --model lenet --epochs 50 --seed 1

# Train MLP on MNIST, single-shot NGF (keep 80%), critical period epochs 5-14
python main.py --task supervised --dataset mnist --model mlp --pruning-method ngf \
    --critical-start-epoch 5 --critical-duration 10 --pruning-threshold 0.8 --epochs 50 --seed 2

# Train LeNet5 on FashionMNIST, magnitude prune 30% at epoch 15
python main.py --task supervised --dataset fashion_mnist --model lenet --pruning-method magnitude \
    --pruning-apply-epoch 15 --pruning-threshold 0.3 --epochs 50 --seed 3

# Train LeNet5 on FashionMNIST, scheduled NGF
python main.py --task supervised --dataset fashion_mnist --model lenet --pruning-method ngf_scheduled \
    --pruning-start-epoch 5 --pruning-end-epoch 35 --pruning-frequency 5 \
    --initial-keep-fraction 1.0 --final-keep-fraction 0.3 --schedule-power 3 \
    --epochs 50 --seed 4

# === Reinforcement Learning ===
# Train PolicyMLP on CartPole, no pruning, save video every 200 episodes
python main.py --task rl --env-name CartPole-v1 --episodes 1500 --video-every-n-episodes 200 --seed 10

# Train PolicyMLP on CartPole, single-shot NGF (keep 70%), critical period episodes 100-399
python main.py --task rl --env-name CartPole-v1 --episodes 1500 --pruning-method ngf \
    --critical-start-episode 100 --critical-duration-episodes 300 --pruning-threshold 0.7 --seed 11

# Train PolicyMLP on CartPole, magnitude prune 25% at episode 500
python main.py --task rl --env-name CartPole-v1 --episodes 1500 --pruning-method magnitude \
    --pruning-apply-episode 500 --pruning-threshold 0.25 --seed 12
```

Show all arguments:
```bash
python main.py --help
```

## Running Batch Experiments and Plotting

*   **Supervised:** The script `run_experiments.sh` provides an example of running a grid of *supervised* experiments. You may need to modify it (`chmod +x run_experiments.sh` then `./run_experiments.sh`). It saves results to `results/supervised/...` and then calls `plot_results.py`.
*   **RL:** You can adapt `run_experiments.sh` or create a similar script to run batches of RL experiments by setting `--task rl` and relevant RL/pruning arguments. Results will be saved under `results/rl/...`.
*   **Plotting:**
    *   `RLTrainer` saves basic performance plots (`rl_training_plots.png`) within each RL run's directory.
    *   `plot_results.py` aggregates data *only* from supervised runs (currently hardcoded for a specific dataset like 'fashion_mnist' - see script) found within the `results/supervised/` subdirectories and generates comparison plots (accuracy vs. sparsity, etc.) saved to `--output-dir` (default: `results/comparison_plots/`). To plot comparisons for RL results, this script would need modification. 