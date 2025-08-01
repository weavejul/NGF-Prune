import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.nn.utils.prune as prune
import torch.nn as nn
import numpy as np
import time
import os
import json
import imageio
from tqdm import tqdm
import math
from collections import deque
import pandas as pd

from src.pruning import (
    add_activation_hooks,
    remove_hooks,
    calculate_ngf_masks,
    apply_masks,
    get_model_pruning_rate,
    convert_to_prunable,
    PrunableLinear,
    PrunableConv2d
)
from src.rl_env import get_cartpole_env

class RLTrainer:
    def __init__(self, model, env_name, device, args):
        self.model = model
        self.env_name = env_name
        self.device = device
        self.args = args

        # Env Setup
        self.env = get_cartpole_env(render_mode="rgb_array")
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.gamma = args.gamma

        # Logging
        self.history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'avg_rewards_window': [],
            'pruning_rate': [],
            'episode_times': [],
            'pruning_applied_episode': -1,
            'layer_stats_at_pruning': {},
            'videos_saved': [] 
        }
        self.reward_window = deque(maxlen=100)

        # Video Saving
        self.video_dir = os.path.join(args.results_dir, "videos")
        self.video_save_freq = args.video_every_n_episodes
        os.makedirs(self.video_dir, exist_ok=True)

        # Pruning Setup
        self.pruning_method = args.pruning_method
        self.activation_hooks = None

        if self.pruning_method in ['ngf', 'ngf_scheduled']:
            print("Converting model to prunable layers for NGF...")
            convert_to_prunable(self.model)
            self.model.to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        elif self.pruning_method == 'magnitude':
            print("Magnitude pruning selected. Will be applied during training.")
        else:
            print("No pruning selected.")

        # Scheduled Pruning Params
        if self.pruning_method == 'ngf_scheduled':
             self.pruning_start_episode = args.pruning_start_episode
             self.pruning_end_episode = args.pruning_end_episode
             self.pruning_frequency = args.pruning_frequency
             self.initial_keep_fraction = args.initial_keep_fraction
             self.final_keep_fraction = args.final_keep_fraction
             self.schedule_power = args.schedule_power
             self.total_pruning_steps = math.floor(
                 (self.pruning_end_episode - self.pruning_start_episode) / self.pruning_frequency
             ) if self.pruning_frequency > 0 else 0


    def _zero_pruned_gradients(self):
        """Manually sets gradients of pruned weights/biases to zero."""
        for module in self.model.modules():
            if hasattr(module, 'mask'): # For NGF Peunable layers
                 if module.pruned and module.weight.grad is not None:
                     if isinstance(module, PrunableLinear):
                         mask_shape_weight = module.mask.unsqueeze(1)
                         mask_shape_weight = module.mask.view(-1, 1, 1, 1)
                     else: continue
                     module.weight.grad.data *= mask_shape_weight
                 if module.pruned and module.bias is not None and module.bias.grad is not None:
                     module.bias.grad.data *= module.mask
            elif isinstance(module, (nn.Linear, nn.Conv2d)) and prune.is_pruned(module):
                # For torch.nn.utils.prune
                if hasattr(module, 'weight_mask') and module.weight.grad is not None:
                    module.weight.grad.data *= module.weight_mask
                if hasattr(module, 'bias_mask') and module.bias is not None and module.bias.grad is not None:
                    module.bias.grad.data *= module.bias_mask

    def _save_episode_video(self, episode_frames, episode_num):
        """Saves collected frames as a video."""
        if not episode_frames:
            return
        video_path = os.path.join(self.video_dir, f"episode_{episode_num}.mp4")
        try:
            imageio.mimsave(video_path, episode_frames, fps=30)
            self.history['videos_saved'].append(episode_num)
            print(f"Saved video for episode {episode_num} to {video_path}")
        except Exception as e:
            print(f"Error saving video for episode {episode_num}: {e}")

    def select_action(self, state):
        """Selects an action using the current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action_logits = self.model(state)
        probs = F.softmax(action_logits, dim=-1)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item(), log_prob

    def calculate_returns(self, rewards):
        """Calculates discounted returns."""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update_policy(self, log_probs, returns):
        """Performs the policy update."""
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()

        self._zero_pruned_gradients()

        self.optimizer.step()


    def apply_ngf_pruning(self, episode, current_keep_fraction):
        """Applies NGF pruning (episode-based version)."""
        print(f"Episode {episode}: Calculating and applying NGF masks (Keep Fraction: {current_keep_fraction:.4f})...")
        masks_dict, overall_pruning_rate, layer_stats = calculate_ngf_masks(self.model, current_keep_fraction)

        applied_stats = {}
        actual_final_rate = get_model_pruning_rate(self.model)

        if masks_dict is not None:
            apply_masks(self.model, masks_dict)
            self.history['pruning_applied_episode'] = episode
            actual_final_rate = get_model_pruning_rate(self.model)
            applied_stats = layer_stats
            print(f"NGF pruning applied. New overall pruning rate: {actual_final_rate*100:.2f}%")
            print("Resetting optimizer state after pruning step.")
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        else:
            print("NGF pruning mask calculation failed.")

        self.history['layer_stats_at_pruning'] = applied_stats
        return actual_final_rate

    def apply_magnitude_pruning(self, episode):
        """Applies magnitude pruning (episode-based version)."""
        print(f"Episode {episode}: Applying magnitude pruning...")
        layer_stats = {}
        parameters_to_prune = []
        prunable_modules = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))
                prunable_modules[name] = module

        if not parameters_to_prune:
            print("No Linear or Conv2d layers found for magnitude pruning.")
            current_pruning_rate = 0.0
            if self.history['pruning_rate']:
                 current_pruning_rate = self.history['pruning_rate'][-1]
            self.history['pruning_rate'].append(current_pruning_rate)
            self.history['layer_stats_at_pruning'] = {}
            return current_pruning_rate

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.args.pruning_threshold,
        )

        total_params = 0
        pruned_params = 0
        for name, module in prunable_modules.items():
             prune.remove(module, 'weight')
             layer_total = module.weight.nelement()
             layer_pruned = torch.sum(module.weight == 0).item()
             layer_pruning_rate = layer_pruned / layer_total if layer_total > 0 else 0.0
             layer_stats[name] = {
                 'total_params': layer_total,
                 'pruned_params': layer_pruned,
                 'pruning_rate': layer_pruning_rate
             }
             total_params += layer_total
             pruned_params += layer_pruned
             print(f"Layer '{name}' (Magnitude): Pruned {layer_pruned}/{layer_total} weights ({layer_pruning_rate*100:.2f}%)")

        final_pruning_rate = pruned_params / total_params if total_params > 0 else 0.0
        self.history['pruning_applied_episode'] = episode
        self.history['layer_stats_at_pruning'] = layer_stats
        print(f"Magnitude pruning applied and made permanent. Overall weight pruning rate: {final_pruning_rate*100:.2f}%")
        print("Resetting optimizer state after pruning.")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return final_pruning_rate

    def get_scheduled_keep_fraction(self, episode):
        """Calculates the keep fraction for the current episode (scheduled NGF)."""
        if episode < self.pruning_start_episode:
            return 1.0
        if episode >= self.pruning_end_episode:
             return self.final_keep_fraction

        current_step = math.floor((episode - self.pruning_start_episode) / self.pruning_frequency)
        total_steps = self.total_pruning_steps
        progress = current_step / total_steps if total_steps > 0 else 1.0
        progress = min(progress, 1.0)

        keep_fraction = self.final_keep_fraction + \
                        (self.initial_keep_fraction - self.final_keep_fraction) * \
                        math.pow(1.0 - progress, self.schedule_power)

        keep_fraction = max(self.final_keep_fraction, min(self.initial_keep_fraction, keep_fraction))
        return keep_fraction


    def train(self):
        num_episodes = self.args.episodes
        pruning_done = False
        hooks_active = False
        self.activation_hooks = None

        # Pruning Schedule Timing
        ngf_critical_start = getattr(self.args, 'critical_start_episode', 0) if self.pruning_method == 'ngf' else -1
        ngf_critical_duration = getattr(self.args, 'critical_duration_episodes', 0) if self.pruning_method == 'ngf' else -1
        ngf_single_shot_prune_episode = ngf_critical_start + ngf_critical_duration if ngf_critical_start >= 0 and ngf_critical_duration > 0 else -1
        magnitude_pruning_episode = getattr(self.args, 'pruning_apply_episode', -1) if self.pruning_method == 'magnitude' else -1

        # Initial Hook Setup
        if self.pruning_method == 'ngf_scheduled' and hasattr(self.args, 'pruning_start_episode') and self.args.pruning_start_episode >= 0:
             print(f"Preparing for scheduled pruning: Adding activation hooks before episode {self.args.pruning_start_episode}.")
             for module in self.model.modules():
                 if hasattr(module, 'reset_activation_stats'): module.reset_activation_stats()
             self.activation_hooks = add_activation_hooks(self.model)
             hooks_active = True
        elif self.pruning_method == 'ngf' and ngf_critical_start >= 0:
             if ngf_critical_start == 0:
                 print(f"Preparing for single-shot NGF: Adding activation hooks at start (Crit Period: {ngf_critical_start}-{ngf_single_shot_prune_episode-1}).")
                 for module in self.model.modules():
                      if hasattr(module, 'reset_activation_stats'): module.reset_activation_stats()
                 self.activation_hooks = add_activation_hooks(self.model)
                 hooks_active = True

        # Training Loop
        progress_bar = tqdm(range(num_episodes), desc="RL Training Progress")
        for i_episode in progress_bar:
            state, _ = self.env.reset()
            log_probs = []
            rewards = []
            frames = []
            save_video_this_episode = (self.video_save_freq > 0 and i_episode % self.video_save_freq == 0)

            # Pruning Logic within the loop
            current_overall_rate = get_model_pruning_rate(self.model) if self.pruning_method != 'none' else 0.0
            current_keep_fraction = 1.0 # Default if not scheduled NGF

            # Single-Shot NGF Pruning
            if self.pruning_method == 'ngf':
                 if i_episode == ngf_critical_start and not hooks_active:
                     print(f"Episode {i_episode}: Entering single-shot NGF critical period ({ngf_critical_duration} episodes). Adding activation hooks.")
                     for module in self.model.modules():
                          if hasattr(module, 'reset_activation_stats'): module.reset_activation_stats()
                     self.activation_hooks = add_activation_hooks(self.model)
                     hooks_active = True

                 if i_episode == ngf_single_shot_prune_episode and not pruning_done:
                      if hooks_active:
                          print(f"Episode {i_episode}: End of critical period. Removing hooks before pruning.")
                          remove_hooks(self.activation_hooks)
                          hooks_active = False
                      else: print(f"Warning: Reached NGF prune episode {i_episode}, but hooks inactive!")

                      print(f"--- Single-Shot NGF Pruning Step at Episode {i_episode} ---")
                      ngf_keep_fraction = getattr(self.args, 'pruning_threshold', 0.5)
                      current_overall_rate = self.apply_ngf_pruning(i_episode, ngf_keep_fraction)
                      pruning_done = True

            # Scheduled NGF Pruning
            elif self.pruning_method == 'ngf_scheduled':
                 is_pruning_phase = self.pruning_start_episode <= i_episode < self.pruning_end_episode
                 should_prune_this_episode = is_pruning_phase and ((i_episode - self.pruning_start_episode) % self.pruning_frequency == 0)

                 if should_prune_this_episode:
                      if not hooks_active: print(f"Warning: Attempting scheduled prune ep {i_episode} but hooks inactive!")
                      else:
                          current_keep_fraction = self.get_scheduled_keep_fraction(i_episode)
                          print(f"-- Scheduled Pruning Step {i_episode} (Keep Frac: {current_keep_fraction:.4f}) --")
                          current_overall_rate = self.apply_ngf_pruning(i_episode, current_keep_fraction)
                          print(f"Episode {i_episode}: Resetting activation stats AFTER pruning.")
                          for module in self.model.modules():
                              if hasattr(module, 'reset_activation_stats'): module.reset_activation_stats()

                 # Remove hooks at the end of schedule
                 is_pruning_phase_next = self.pruning_start_episode <= (i_episode + 1) < self.pruning_end_episode
                 if not is_pruning_phase_next and hooks_active:
                      print(f"End of Episode {i_episode}: Exiting scheduled NGF pruning phase. Removing hooks.")
                      remove_hooks(self.activation_hooks)
                      hooks_active = False

            # Magnitude Pruning
            elif self.pruning_method == 'magnitude' and not pruning_done and i_episode == magnitude_pruning_episode:
                 print("--- Magnitude Pruning Step ---")
                 current_overall_rate = self.apply_magnitude_pruning(i_episode)
                 pruning_done = True

            # Rollout Episode
            start_time = time.time()
            self.model.eval() # Hooks still work
            episode_reward = 0
            for t in range(1, self.env._max_episode_steps + 1):
                action, log_prob = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                log_probs.append(log_prob)
                rewards.append(reward)
                episode_reward += reward

                if save_video_this_episode:
                    frame = self.env.render()
                    if frame is not None:
                         frames.append(frame)

                state = next_state
                if done:
                    break
            self.model.train()

            # Record History
            self.history['episode_rewards'].append(episode_reward)
            self.history['episode_lengths'].append(t)
            self.history['pruning_rate'].append(current_overall_rate)
            self.reward_window.append(episode_reward)
            avg_reward = np.mean(self.reward_window)
            self.history['avg_rewards_window'].append(avg_reward)
            episode_time = time.time() - start_time
            self.history['episode_times'].append(episode_time)

            # Update Policy
            returns = self.calculate_returns(rewards)
            self.update_policy(log_probs, returns)

            # Save Video
            if save_video_this_episode:
                self._save_episode_video(frames, i_episode)

            # Logging
            progress_bar.set_postfix(
                LastReward=f"{episode_reward:.1f}",
                AvgReward100=f"{avg_reward:.2f}",
                Sparsity=f"{current_overall_rate*100:.1f}%"
            )

        # End of Training
        if hooks_active:
            print(f"End of RL training loop: Removing activation hooks after episode {i_episode}.")
            remove_hooks(self.activation_hooks)

        self.env.close()
        print(f"\nRL Training finished. Final avg reward (last 100 ep): {avg_reward:.2f}")
        return self.history

    def save_results(self, results_dir):
        """Saves training history, config, plots, and final model."""
        os.makedirs(self.video_dir, exist_ok=True)

        # Save config
        config_path = os.path.join(results_dir, 'config.json')
        with open(config_path, 'w') as f:
            config_to_save = vars(self.args).copy()
            # Add RL specific info if not in args already
            config_to_save['env_name'] = self.env_name
            config_to_save['state_dim'] = int(self.state_dim) if isinstance(self.state_dim, np.number) else self.state_dim
            config_to_save['action_dim'] = int(self.action_dim) if isinstance(self.action_dim, np.number) else self.action_dim
            
            # Sanitize the entire config dict just in case
            sanitized_config = {}
            for key, value in config_to_save.items():
                if isinstance(value, np.integer):
                    sanitized_config[key] = int(value)
                elif isinstance(value, np.floating):
                    sanitized_config[key] = float(value)
                elif isinstance(value, np.ndarray):
                     sanitized_config[key] = value.tolist()
                else:
                    sanitized_config[key] = value
            
            json.dump(sanitized_config, f, indent=2)
            print(f"Saved config to {config_path}")

        # Save history
        results_path = os.path.join(results_dir, 'results.json')
        serializable_history = {}
        for key, value in self.history.items():
             if isinstance(value, deque):
                  serializable_history[key] = list(value)
             elif key == 'layer_stats_at_pruning':
                 try:
                      sanitized_layer_stats = json.loads(json.dumps(value, default=lambda o: int(o) if isinstance(o, np.integer) else float(o) if isinstance(o, np.floating) else o.tolist() if isinstance(o, np.ndarray) else str(o)))
                      serializable_history[key] = sanitized_layer_stats
                 except Exception as e: # Catch more errors during sanitization
                      print(f"Warning: Could not fully serialize 'layer_stats_at_pruning' due to {e}. Storing as string.")
                      serializable_history[key] = str(value) # Fallback to string
             elif isinstance(value, list):
                  # Apply sanitization to list elements
                  sanitized_list = []
                  for item in value:
                      if isinstance(item, np.integer):
                           sanitized_list.append(int(item))
                      elif isinstance(item, np.floating):
                           sanitized_list.append(float(item))
                      elif isinstance(item, torch.Tensor):
                           sanitized_list.append(item.item() if item.numel() == 1 else item.tolist())
                      elif isinstance(item, np.ndarray):
                           sanitized_list.append(item.tolist())
                      else:
                           sanitized_list.append(item)
                  serializable_history[key] = sanitized_list
             elif isinstance(value, np.integer):
                  serializable_history[key] = int(value)
             elif isinstance(value, np.floating):
                  serializable_history[key] = float(value)
             elif isinstance(value, torch.Tensor):
                  serializable_history[key] = value.item() if value.numel() == 1 else value.tolist()
             elif isinstance(value, np.ndarray):
                  serializable_history[key] = value.tolist()
             else:
                  serializable_history[key] = value # Assume serializable

        try:
             with open(results_path, 'w') as f:
                  json.dump(serializable_history, f, indent=2)
             print(f"Saved results history to {results_path}")
        except TypeError as e:
            print(f"Error serializing RL results to JSON: {e}")



        # Save plots
        self.plot_results(results_dir)

        # Save final model state
        model_path = os.path.join(results_dir, 'final_model.pth')
        torch.save(self.model.state_dict(), model_path)
        print(f"Saved final model to {model_path}")

    def plot_results(self, results_dir):
        """Generates simple plots for RL training."""
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np

        num_episodes_recorded = len(self.history['episode_rewards'])
        if num_episodes_recorded < 2:
            print("Not enough data (less than 2 episodes) to generate plots.")
            return

        episodes_xaxis = np.arange(1, num_episodes_recorded)
        
        rewards_series = pd.Series(self.history['episode_rewards'][1:])
        lengths_series = pd.Series(self.history['episode_lengths'][1:])
        times_series = pd.Series(self.history['episode_times'][1:])
        pruning_rate_pct = [p * 100 for p in self.history['pruning_rate'][1:]]
        
        rolling_window = 100
        min_periods = max(1, rolling_window // 10)

        # Rolling means and stds
        rewards_mean = rewards_series.rolling(window=rolling_window, min_periods=1).mean()
        rewards_std = rewards_series.rolling(window=rolling_window, min_periods=min_periods).std().fillna(0)
        
        lengths_mean = lengths_series.rolling(window=rolling_window, min_periods=1).mean()
        lengths_std = lengths_series.rolling(window=rolling_window, min_periods=min_periods).std().fillna(0)
        
        times_mean = times_series.rolling(window=rolling_window, min_periods=1).mean()
        times_std = times_series.rolling(window=rolling_window, min_periods=min_periods).std().fillna(0)

        plt.figure(figsize=(14, 12))

        # Rewards Plot
        plt.subplot(2, 2, 1)
        plt.plot(episodes_xaxis, rewards_series, label='Raw Ep. Reward', alpha=0.3, color='grey') 
        plt.plot(episodes_xaxis, rewards_mean, label=f'Avg Reward ({rolling_window} ep)', color='blue')
        plt.fill_between(episodes_xaxis, rewards_mean - rewards_std, rewards_mean + rewards_std, 
                         color='blue', alpha=0.2, label=f'±1 Std Dev ({rolling_window} ep)')
        plt.xlabel('Episodes (Starting from 2nd)')
        plt.ylabel('Reward')
        plt.title(f'{self.env_name} Rewards (Mean ± Std Dev over {rolling_window} Episodes)')
        plt.legend(loc='lower right')
        plt.grid(True)

        # Episode Lengths Plot
        plt.subplot(2, 2, 2)
        plt.plot(episodes_xaxis, lengths_mean, label=f'Avg Length ({rolling_window} ep)', color='green')
        plt.fill_between(episodes_xaxis, lengths_mean - lengths_std, lengths_mean + lengths_std, 
                         color='green', alpha=0.2, label=f'±1 Std Dev ({rolling_window} ep)')
        plt.xlabel('Episodes (Starting from 2nd)')
        plt.ylabel('Steps')
        plt.title(f'Episode Lengths (Mean ± Std Dev over {rolling_window} Episodes)')
        plt.legend(loc='lower right')
        plt.grid(True)

        # Pruning Rate Plot
        plt.subplot(2, 2, 3)
        plt.plot(episodes_xaxis, pruning_rate_pct, label='Pruning Rate (%)')
        if self.history['pruning_applied_episode'] != -1 and self.history['pruning_applied_episode'] > 0:
            plt.axvline(x=self.history['pruning_applied_episode'], color='r', linestyle='--',
                        label=f'Pruning @ Ep {self.history["pruning_applied_episode"]}')
        plt.xlabel('Episodes (Starting from 2nd)')
        plt.ylabel('Pruning Rate (%)')
        plt.title('Model Pruning Rate Over Training')
        plt.ylim(-5, 105)
        plt.legend()
        plt.grid(True)

        # Episode Times Plot
        plt.subplot(2, 2, 4)
        plt.plot(episodes_xaxis, times_mean, label=f'Avg Time ({rolling_window} ep)', color='red')
        plt.fill_between(episodes_xaxis, times_mean - times_std, times_mean + times_std, 
                         color='red', alpha=0.2, label=f'±1 Std Dev ({rolling_window} ep)')
        plt.xlabel('Episodes (Starting from 2nd)')
        plt.ylabel('Time (s)')
        plt.title(f'Time Per Episode (Mean ± Std Dev over {rolling_window} Episodes)')
        plt.legend(loc='upper left')
        plt.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(results_dir, 'rl_training_plots.png')
        plt.savefig(plot_path)
        print(f"RL training plots saved to {plot_path}")
        plt.close() 