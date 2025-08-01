import argparse
import torch
import json
import os
import time
import numpy as np

from src.data_loader import get_mnist_loaders, get_fashion_mnist_loaders
from src.models import SimpleMLP, LeNet5, PolicyMLP
from src.trainer import Trainer
from src.rl_env import get_cartpole_env
from src.rl_trainer import RLTrainer
from src.pruning import get_model_pruning_rate, calculate_weight_sparsity
from src.constants import (
    SUPPORTED_DATASETS, SUPPORTED_MODELS, ERROR_INVALID_DATASET, 
    ERROR_INVALID_MODEL, ERROR_INVALID_TASK, ERROR_INVALID_KEEP_FRACTION,
    ERROR_INVALID_PRUNE_FRACTION, ERROR_INVALID_CRITICAL_PERIOD,
    ERROR_INVALID_APPLY_EPOCH, ERROR_INVALID_SCHEDULE_FRACTIONS,
    ERROR_INVALID_SCHEDULE_EPOCHS, ERROR_INVALID_PRUNING_FREQUENCY
)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    set_seed(args.seed)
    print("Starting experiment with config:")
    print(json.dumps(vars(args), indent=2))

    # Setup device
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: {args.gpu}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Create results directory
    method_details = args.pruning_method
    if args.task == 'rl':
        task_identifier = f"rl_{args.env_name}"
        model_identifier = "policy_mlp"
        if args.pruning_method == 'ngf':
             method_details = f"ngf_start{args.critical_start_episode}_dur{args.critical_duration_episodes}_keep{args.pruning_threshold:.2f}"
        elif args.pruning_method == 'magnitude':
             method_details = f"mag_ep{args.pruning_apply_episode}_prune{args.pruning_threshold:.2f}"
    else: # Supervised task
        task_identifier = args.dataset
        model_identifier = args.model
        if args.pruning_method == 'ngf':
             method_details = f"ngf_start{args.critical_start_epoch}_dur{args.critical_duration}_keep{args.pruning_threshold:.2f}"
        elif args.pruning_method == 'magnitude':
             method_details = f"mag_epoch{args.pruning_apply_epoch}_prune{args.pruning_threshold:.2f}"

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_name = f"{task_identifier}_{model_identifier}_{method_details}_seed{args.seed}_{timestamp}"
    task_type = 'rl' if args.task == 'rl' else 'supervised'
    results_dir = os.path.join(args.results_dir, task_type, task_identifier, model_identifier, run_name)
    args.results_dir = results_dir # Update args to contain full path
    print(f"Results will be saved to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)


    # Task specific setup
    if args.task == 'rl':
        print(f"Setting up RL task: {args.env_name}")
        # Init model
        env_temp = get_cartpole_env() # Create temporary env to get dims
        state_dim = env_temp.observation_space.shape[0]
        action_dim = env_temp.action_space.n
        env_temp.close()
        model = PolicyMLP(input_dim=state_dim, output_dim=action_dim)
        model.to(device)
        print(model)

        # Init RL trainer
        print("Initializing RLTrainer...")
        trainer = RLTrainer(
            model=model,
            env_name=args.env_name,
            device=device,
            args=args
        )
        print("RLTrainer initialized.")

        # Run RL training
        print("Starting RL training...")
        start_time = time.time()
        results = trainer.train()
        end_time = time.time()
        training_duration = end_time - start_time
        print(f"RL Training finished in {training_duration:.2f} seconds")

        # Add final metrics to results dict before saving
        results['total_training_time'] = training_duration
        final_avg_reward = results['avg_rewards_window'][-1] if results['avg_rewards_window'] else None
        results['final_avg_reward_100ep'] = final_avg_reward
        results['final_pruning_rate'] = results['pruning_rate'][-1] if results['pruning_rate'] else 0.0

        final_model = trainer.model
        if args.pruning_method in ['ngf', 'ngf_scheduled']:
            sparsity_type = "Mask (NGF Neuron/Channel)"
        elif args.pruning_method == 'magnitude':
            sparsity_type = "Weight (Magnitude)"
        else: # 'none'
             sparsity_type = "None"
        print(f"Final Model Sparsity ({sparsity_type}): {results['final_pruning_rate']:.4f}")

        # Save results
        print(f"Saving RL results to {results_dir}...")
        trainer.save_results(results_dir)
        print("RL Experiment finished successfully.")


    elif args.task == 'supervised':
        print("Setting up Supervised task...")
        # Load data
        print(f"Loading {args.dataset} dataset...")
        if args.dataset == 'mnist':
            train_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size)
        elif args.dataset == 'fashion_mnist':
            train_loader, test_loader = get_fashion_mnist_loaders(batch_size=args.batch_size)
        else:
            raise ValueError(ERROR_INVALID_DATASET.format(args.dataset))
        print("Data loaded.")

        # Init model
        print(f"Initializing model: {args.model}")
        if args.model == 'mlp':
            model = SimpleMLP()
        elif args.model == 'lenet':
            model = LeNet5()
        else:
            raise ValueError(ERROR_INVALID_MODEL.format(args.model))
        model.to(device)
        print(model)

        # Init trainer
        print("Initializing Trainer...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            args=args
        )
        print("Trainer initialized.")

        # Run training
        print("Starting training...")
        start_time = time.time()
        results = trainer.train()
        end_time = time.time()
        training_duration = end_time - start_time
        print(f"Training finished in {training_duration:.2f} seconds")

        # Save results
        print(f"Saving supervised results to {results_dir}...")
        results['total_training_time'] = training_duration
        results['final_accuracy'] = results['test_accuracy'][-1] if 'test_accuracy' in results and results['test_accuracy'] else None
        results['best_accuracy'] = trainer.best_test_accuracy

        final_model = trainer.model
        if args.pruning_method in ['ngf', 'ngf_scheduled']:
            final_sparsity = get_model_pruning_rate(final_model)
            sparsity_type = "Mask (NGF)"
        elif args.pruning_method == 'magnitude':
            final_sparsity = calculate_weight_sparsity(final_model)
            sparsity_type = "Weight (Magnitude)"
        else:
            final_sparsity = 0.0
            sparsity_type = "None"

        results['final_pruning_rate'] = final_sparsity
        print(f"Final Model Sparsity ({sparsity_type}): {final_sparsity:.4f}")

        trainer.save_results(results_dir)
        print("Supervised Experiment finished successfully.")

    else:
        raise ValueError(ERROR_INVALID_TASK.format(args.task))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NGF-Inspired Pruning Experiments (Supervised & RL)')

    # General
    parser.add_argument('--task', type=str, default='supervised', choices=['supervised', 'rl'], help='Task type to run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID to use (default: None, use CPU)')
    parser.add_argument('--results-dir', type=str, default='results', help='Base directory to save results (task/dataset/model structure will be created within)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='Learning rate (default: 0.001)')

    # Supervised task specific
    sup_group = parser.add_argument_group('Supervised Task Arguments')
    sup_group.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist'], help='Dataset to use (supervised only)')
    sup_group.add_argument('--model', type=str, default='mlp', choices=['mlp', 'lenet'], help='Model architecture (supervised only)')
    sup_group.add_argument('--batch-size', type=int, default=128, metavar='N', help='Input batch size for training (supervised only)')
    sup_group.add_argument('--epochs', type=int, default=50, metavar='N', help='Maximum number of epochs to train (supervised only)')
    sup_group.add_argument('--early-stopping-patience', type=int, default=4, help='Epochs to wait for improvement before stopping (supervised only)')

    # RL task specific
    rl_group = parser.add_argument_group('Reinforcement Learning Task Arguments')
    rl_group.add_argument('--env-name', type=str, default='CartPole-v1', help='Gymnasium environment name (rl only)')
    rl_group.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train (rl only)')
    rl_group.add_argument('--gamma', type=float, default=0.99, help='Discount factor for REINFORCE (rl only)')
    rl_group.add_argument('--video-every-n-episodes', type=int, default=100, help='Frequency to save agent gameplay videos (0 to disable, rl only)')

    # Pruning shared
    prune_group = parser.add_argument_group('Pruning Arguments (Shared)')
    prune_group.add_argument('--pruning-method', type=str, default='none', choices=['none', 'ngf', 'magnitude', 'ngf_scheduled'], help='Pruning method')
    # Pruning threshold meaning depends on method AND task (keep fraction for NGF, prune fraction for Magnitude)
    prune_group.add_argument('--pruning-threshold', type=float, default=0.7, help='NGF: Fraction of neurons/channels to KEEP | Magnitude: Fraction of weights to PRUNE')

    # Single-Shot Pruning Timing
    timing_group = parser.add_argument_group('Single-Shot Pruning Timing')
    # NGF (Supervised)
    timing_group.add_argument('--critical-start-epoch', type=int, default=5, help='Epoch to start critical period (NGF supervised)')
    timing_group.add_argument('--critical-duration', type=int, default=10, help='Duration of critical period (NGF supervised)')
    # NGF (RL)
    timing_group.add_argument('--critical-start-episode', type=int, default=100, help='Episode to start critical period (NGF RL)')
    timing_group.add_argument('--critical-duration-episodes', type=int, default=200, help='Duration of critical period in episodes (NGF RL)')
    # Magnitude (Supervised)
    timing_group.add_argument('--pruning-apply-epoch', type=int, default=15, help='Epoch to apply magnitude pruning (Magnitude supervised)')
    # Magnitude (RL)
    timing_group.add_argument('--pruning-apply-episode', type=int, default=300, help='Episode to apply magnitude pruning (Magnitude RL)')

    # Scheduled NGF Pruning Specific
    sched_group = parser.add_argument_group('Scheduled NGF Pruning Arguments')
    sched_group.add_argument('--pruning-start-epoch', type=int, default=5, help='Epoch/Episode to start scheduled pruning') # Name is generic now
    sched_group.add_argument('--pruning-end-epoch', type=int, default=30, help='Epoch/Episode to end scheduled pruning (pruning stops *before* this)') # Name is generic
    sched_group.add_argument('--pruning-frequency', type=int, default=2, help='Frequency (in epochs/episodes) to apply pruning step')
    sched_group.add_argument('--initial-keep-fraction', type=float, default=1.0, help='Initial keep fraction at start of schedule')
    sched_group.add_argument('--final-keep-fraction', type=float, default=0.2, help='Final target keep fraction at end of schedule')
    sched_group.add_argument('--schedule-power', type=float, default=3.0, help='Exponent for the polynomial schedule (e.g., 3 for cubic)')

    args = parser.parse_args()

    # Validate args
    # Generic validation can happen here, task-specific val below

    if args.task == 'rl':
        # Use episode-based args for validation
        if args.pruning_method == 'ngf':
            if args.critical_start_episode < 0: raise ValueError("Critical start episode must be non-negative.")
            if args.critical_duration_episodes <= 0: raise ValueError(ERROR_INVALID_CRITICAL_PERIOD)
            if not (0.0 < args.pruning_threshold <= 1.0): raise ValueError(ERROR_INVALID_KEEP_FRACTION)
        elif args.pruning_method == 'magnitude':
            if args.pruning_apply_episode <= 0: raise ValueError(ERROR_INVALID_APPLY_EPOCH)
            if not (0.0 <= args.pruning_threshold < 1.0): raise ValueError(ERROR_INVALID_PRUNE_FRACTION)
        elif args.pruning_method == 'ngf_scheduled':
             # Rename args used by RLTrainer to match generic names
             args.pruning_start_episode = args.pruning_start_epoch
             args.pruning_end_episode = args.pruning_end_epoch
             # Validate scheduled params
             if not (0.0 < args.final_keep_fraction <= args.initial_keep_fraction <= 1.0): raise ValueError(ERROR_INVALID_SCHEDULE_FRACTIONS)
             if args.pruning_start_episode < 0 or args.pruning_end_episode <= args.pruning_start_episode: raise ValueError(ERROR_INVALID_SCHEDULE_EPOCHS)
             if args.pruning_frequency <= 0: raise ValueError(ERROR_INVALID_PRUNING_FREQUENCY)

    elif args.task == 'supervised':
        # Use epoch-based args for validation
        if args.pruning_method == 'ngf':
            if args.critical_start_epoch < 0: raise ValueError("Critical start epoch must be non-negative.")
            if args.critical_duration <= 0: raise ValueError(ERROR_INVALID_CRITICAL_PERIOD)
            if not (0.0 < args.pruning_threshold <= 1.0): raise ValueError(ERROR_INVALID_KEEP_FRACTION)
        elif args.pruning_method == 'magnitude':
            if args.pruning_apply_epoch <= 0: raise ValueError(ERROR_INVALID_APPLY_EPOCH)
            if not (0.0 <= args.pruning_threshold < 1.0): raise ValueError(ERROR_INVALID_PRUNE_FRACTION)
        elif args.pruning_method == 'ngf_scheduled':
             # Validate scheduled params
             if not (0.0 < args.final_keep_fraction <= args.initial_keep_fraction <= 1.0): raise ValueError(ERROR_INVALID_SCHEDULE_FRACTIONS)
             if args.pruning_start_epoch < 0 or args.pruning_end_epoch <= args.pruning_start_epoch: raise ValueError(ERROR_INVALID_SCHEDULE_EPOCHS)
             if args.pruning_frequency <= 0: raise ValueError(ERROR_INVALID_PRUNING_FREQUENCY)

    main(args) 