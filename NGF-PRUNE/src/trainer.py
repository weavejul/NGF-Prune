import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import time
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

from src.pruning import (
    add_activation_hooks,
    remove_hooks,
    calculate_ngf_masks,
    apply_masks,
    get_model_pruning_rate,
    convert_to_prunable,
    get_active_neurons,
    get_total_neurons,
    PrunableLinear,
    PrunableConv2d
)

class Trainer:
    def __init__(self, model, train_loader, test_loader, device, args):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.args = args
        self.activation_hooks = None

        self.optimizer = optim.Adam(model.parameters(), lr=args.lr)
        self.criterion = nn.CrossEntropyLoss()

        self.history = {
            'train_loss': [],
            'test_loss': [],
            'test_accuracy': [],
            'pruning_rate': [],
            'epoch_times': [],
            'critical_period_active': [],
            'pruning_applied_epoch': -1,
            'layer_stats_at_pruning': {},
            'keep_fraction_schedule': []
        }
        self.best_test_accuracy = 0.0
        self.epochs_no_improve = 0
        self.early_stopping_patience = args.early_stopping_patience

        self.pruning_method = args.pruning_method

        if self.pruning_method == 'ngf': # Original single-shot NGF
            print("Converting model to prunable layers for Single-Shot NGF...")
            convert_to_prunable(self.model)
            self.model.to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        elif self.pruning_method == 'ngf_scheduled':
            print("Converting model to prunable layers for Scheduled NGF...")
            convert_to_prunable(self.model)
            self.model.to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
            self.total_pruning_steps = math.floor((args.pruning_end_epoch - args.pruning_start_epoch) / args.pruning_frequency)
        elif self.pruning_method == 'magnitude':
            print("Magnitude pruning selected. Will be applied later.")
        elif self.pruning_method == 'none':
             print("No pruning selected.")
        else:
             print(f"Warning: Unknown pruning method '{self.pruning_method}' specified in args.")

    def train_epoch(self, epoch, activation_hooks=None):
        self.model.train()
        total_loss = 0
        start_time = time.time()
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()

            # Make sure gradients of pruned weights/biases are zero
            self._zero_pruned_gradients()

            self.optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.train_loader)
        epoch_time = time.time() - start_time

        self.history['train_loss'].append(avg_loss)
        self.history['epoch_times'].append(epoch_time)
        self.history['critical_period_active'].append(False)

        print(f"Epoch {epoch} [Train]: Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
        return activation_hooks

    def test_epoch(self, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            progress_bar = tqdm(self.test_loader, desc=f"Epoch {epoch} [Test]")
            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                batch_loss = self.criterion(output, target).item()
                test_loss += batch_loss
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)
                progress_bar.set_postfix(accuracy=f"{100. * correct / total:.2f}%")

        # Loss per batch
        avg_loss = test_loss / len(self.test_loader)
        accuracy = 100. * correct / total

        self.history['test_loss'].append(avg_loss)
        self.history['test_accuracy'].append(accuracy)

        print(f"Epoch {epoch} [Test]: Avg Loss: {avg_loss:.6f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)")
        return accuracy

    def _zero_pruned_gradients(self):
        """Manually sets gradients of pruned weights/biases to zero."""
        for module in self.model.modules():
            if hasattr(module, 'mask'):
                 if module.pruned and module.weight.grad is not None:
                     if isinstance(module, PrunableLinear):
                         mask_shape_weight = module.mask.unsqueeze(1)
                     elif isinstance(module, PrunableConv2d):
                         mask_shape_weight = module.mask.view(-1, 1, 1, 1)
                     else:
                         continue
                     module.weight.grad.data *= mask_shape_weight
                 if module.pruned and module.bias is not None and module.bias.grad is not None:
                     module.bias.grad.data *= module.mask
            elif isinstance(module, (nn.Linear, nn.Conv2d)) and prune.is_pruned(module):
                mask = module.weight_mask
                if module.weight.grad is not None:
                    module.weight.grad.data *= mask
                if hasattr(module, 'bias_mask') and module.bias is not None and module.bias.grad is not None:
                    module.bias.grad.data *= module.bias_mask


    def apply_ngf_pruning(self, epoch, current_keep_fraction):
        """Applies NGF pruning with a specific keep fraction."""
        print(f"Epoch {epoch}: Calculating and applying NGF masks (Keep Fraction: {current_keep_fraction:.4f})...")
        masks_dict, overall_pruning_rate, layer_stats = calculate_ngf_masks(self.model, current_keep_fraction)
        
        applied_stats = {}
        actual_final_rate = get_model_pruning_rate(self.model)

        if masks_dict is not None:
            apply_masks(self.model, masks_dict)
            self.history['pruning_applied_epoch'] = epoch
            actual_final_rate = get_model_pruning_rate(self.model)
            applied_stats = layer_stats
            print(f"NGF pruning applied. New overall pruning rate: {actual_final_rate*100:.2f}%")
            # Resetting seems easier than checking if apply_masks changed anything
            print("Resetting optimizer state after pruning step.")
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        else:
            print("NGF pruning mask calculation failed.")

        self.history['layer_stats_at_pruning'] = applied_stats 
        return actual_final_rate 

    def apply_magnitude_pruning(self, epoch):
        print(f"Epoch {epoch}: Applying magnitude pruning...")
        layer_stats = {}
        parameters_to_prune = []
        prunable_modules = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))
                prunable_modules[name] = module

        if not parameters_to_prune:
            print("No Linear or Conv2d layers found for magnitude pruning.")
            self.history['pruning_rate'].append(0.0)
            self.history['layer_stats_at_pruning'] = {}
            return 0.0

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
             layer_active = layer_total - layer_pruned
             layer_pruning_rate = layer_pruned / layer_total if layer_total > 0 else 0.0

             layer_stats[name] = {
                 'total_params': layer_total,
                 'pruned_params': layer_pruned,
                 'active_params': layer_active,
                 'pruning_rate': layer_pruning_rate
             }
             total_params += layer_total
             pruned_params += layer_pruned
             print(f"Layer '{name}' (Magnitude): Pruned {layer_pruned}/{layer_total} weights ({layer_pruning_rate*100:.2f}%)")

        final_pruning_rate = pruned_params / total_params if total_params > 0 else 0.0
        self.history['pruning_applied_epoch'] = epoch
        self.history['layer_stats_at_pruning'] = layer_stats
        print(f"Magnitude pruning applied and made permanent. Overall weight pruning rate: {final_pruning_rate*100:.2f}%")

        # Reset optimizer state
        print("Resetting optimizer state after pruning.")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return final_pruning_rate

    def check_early_stopping(self, current_accuracy, epoch):
        if current_accuracy > self.best_test_accuracy:
            self.best_test_accuracy = current_accuracy
            self.epochs_no_improve = 0
            # torch.save(self.model.state_dict(), "best_model.pth")
        else:
            self.epochs_no_improve += 1
            print(f"Epoch {epoch}: No improvement in test accuracy for {self.epochs_no_improve} epochs.")

        if self.epochs_no_improve >= self.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch} due to no improvement for {self.early_stopping_patience} epochs.")
            return True
        return False

    def get_scheduled_keep_fraction(self, epoch):
        """Calculates the keep fraction for the current epoch based on polynomial decay."""
        args = self.args
        if epoch < args.pruning_start_epoch:
            return 1.0 # Keep everything before schedule starts
        if epoch > args.pruning_end_epoch:
             return args.final_keep_fraction
        
        current_step = math.floor((epoch - args.pruning_start_epoch) / args.pruning_frequency)
        total_steps = self.total_pruning_steps
        
        progress = current_step / total_steps if total_steps > 0 else 1.0
        progress = min(progress, 1.0)

        # Polynomial decay
        keep_fraction = args.final_keep_fraction + \
                        (args.initial_keep_fraction - args.final_keep_fraction) * \
                        math.pow(1.0 - progress, args.schedule_power)
        
        keep_fraction = max(args.final_keep_fraction, min(args.initial_keep_fraction, keep_fraction))
        
        return keep_fraction

    def train(self):
        pruning_done = False
        hooks_active = False
        self.activation_hooks = None

        # Single-shot NGF/Magnitude params
        ngf_critical_start = getattr(self.args, 'critical_start_epoch', 0) if self.pruning_method == 'ngf' else -1
        ngf_critical_duration = getattr(self.args, 'critical_duration', 0) if self.pruning_method == 'ngf' else -1
        ngf_single_shot_prune_epoch = ngf_critical_start + ngf_critical_duration if ngf_critical_start >= 0 and ngf_critical_duration > 0 else -1
        magnitude_pruning_epoch = getattr(self.args, 'pruning_apply_epoch', -1) if self.pruning_method == 'magnitude' else -1

        # Initial hook setup for scheduled or single-shot NGF
        if self.pruning_method == 'ngf_scheduled' and self.args.pruning_start_epoch >= 0:
            print(f"Preparing for scheduled pruning: Adding activation hooks before epoch {self.args.pruning_start_epoch}.")
            for module in self.model.modules():
                if hasattr(module, 'reset_activation_stats'):
                    module.reset_activation_stats()
            self.activation_hooks = add_activation_hooks(self.model)
            hooks_active = True
        # Single-shot NGF hook start
        elif self.pruning_method == 'ngf' and ngf_critical_start >= 0:
            if ngf_critical_start == 0:
                 print(f"Preparing for single-shot NGF: Adding activation hooks at start (Critical Period: {ngf_critical_start}-{ngf_single_shot_prune_epoch-1}).")
                 for module in self.model.modules():
                     if hasattr(module, 'reset_activation_stats'):
                         module.reset_activation_stats()
                 self.activation_hooks = add_activation_hooks(self.model)
                 hooks_active = True
        
        for epoch in range(self.args.epochs):
            current_overall_rate = get_model_pruning_rate(self.model) if self.pruning_method != 'none' else 0.0
            current_keep_fraction = 1.0

            # Single-Shot NGF Hook Management
            if self.pruning_method == 'ngf':
                 if epoch == ngf_critical_start and not hooks_active:
                     print(f"Epoch {epoch}: Entering single-shot NGF critical period ({ngf_critical_duration} epochs). Adding activation hooks.")
                     for module in self.model.modules():
                          if hasattr(module, 'reset_activation_stats'):
                              module.reset_activation_stats()
                     self.activation_hooks = add_activation_hooks(self.model)
                     hooks_active = True
                 
                 if epoch == ngf_single_shot_prune_epoch and not pruning_done:
                      if hooks_active:
                          print(f"Epoch {epoch}: End of critical period. Removing activation hooks before pruning.")
                          remove_hooks(self.activation_hooks)
                          hooks_active = False
                      else:
                          print(f"Warning: Reached NGF pruning epoch {epoch}, but hooks were not active!")
                      
                      print(f"--- Single-Shot NGF Pruning Step at Epoch {epoch} ---")
                      current_overall_rate = self.apply_ngf_pruning(epoch, self.args.pruning_threshold)
                      pruning_done = True

            # Scheduled Pruning Step 
            elif self.pruning_method == 'ngf_scheduled':
                args = self.args
                is_pruning_phase = args.pruning_start_epoch <= epoch < args.pruning_end_epoch
                should_prune_this_epoch = is_pruning_phase and ((epoch - args.pruning_start_epoch) % args.pruning_frequency == 0)
                
                if should_prune_this_epoch:
                    if not hooks_active:
                         print(f"Warning: Attempting scheduled prune at epoch {epoch} but hooks were not active!")
                    else:
                        current_keep_fraction = self.get_scheduled_keep_fraction(epoch)
                        self.history['keep_fraction_schedule'].append({'epoch': epoch, 'keep_fraction': current_keep_fraction})
                        print(f"-- Scheduled Pruning Step {epoch} --")
                        current_overall_rate = self.apply_ngf_pruning(epoch, current_keep_fraction)
                        print(f"Epoch {epoch}: Resetting activation stats AFTER pruning.")
                        for module in self.model.modules():
                            if hasattr(module, 'reset_activation_stats'):
                                module.reset_activation_stats()
                                
            # Magnitude Pruning Step 
            elif self.pruning_method == 'magnitude' and not pruning_done and epoch == magnitude_pruning_epoch:
                 print("--- Magnitude Pruning Step ---")
                 current_overall_rate = self.apply_magnitude_pruning(epoch)
                 pruning_done = True
                 
            # Training Step 
            self.train_epoch(epoch, activation_hooks=None)

            # Record History (Overall Rate)
            self.history['pruning_rate'].append(current_overall_rate)

            # Testing & Early Stopping
            current_accuracy = self.test_epoch(epoch)
            if self.check_early_stopping(current_accuracy, epoch):
                # Remove hooks if stopping early
                if hooks_active:
                     print("Stopping early: Removing activation hooks.")
                     remove_hooks(self.activation_hooks)
                     hooks_active = False
                break

            # End of scheduled NGF phase hook removal
            if self.pruning_method == 'ngf_scheduled':
                 is_pruning_phase_next = args.pruning_start_epoch <= (epoch + 1) < args.pruning_end_epoch
                 if not is_pruning_phase_next and hooks_active: # Remove after the last epoch of the phase
                      print(f"End of Epoch {epoch}: Exiting scheduled NGF pruning phase. Removing activation hooks.")
                      remove_hooks(self.activation_hooks)
                      hooks_active = False

        # Hook cleanup
        if hooks_active:
            print(f"End of training loop: Removing activation hooks after epoch {epoch}.")
            remove_hooks(self.activation_hooks)
            hooks_active = False

        print(f"Training finished. Best test accuracy: {self.best_test_accuracy:.2f}%")
        return self.history

    def save_results(self, results_dir):
        os.makedirs(results_dir, exist_ok=True)

        # Save config
        with open(os.path.join(results_dir, 'config.json'), 'w') as f:
            config_to_save = vars(self.args).copy()
            json.dump(config_to_save, f, indent=2)

        # Save history
        serializable_history = {}
        for key, value in self.history.items():
            if key == 'keep_fraction_schedule':
                 serializable_history[key] = value
            elif key == 'layer_stats_at_pruning':
                serializable_history[key] = value
            elif isinstance(value, list):
                 serializable_history[key] = [item.item() if isinstance(item, (torch.Tensor, np.ndarray)) and item.numel() == 1 else item for item in value]
            elif isinstance(value, (torch.Tensor, np.ndarray)):
                 serializable_history[key] = value.item() if value.numel() == 1 else value.tolist()
            else:
                 serializable_history[key] = value
        results_path = os.path.join(results_dir, 'results.json')
        try:
            with open(results_path, 'w') as f:
                json.dump(serializable_history, f, indent=2)
        except TypeError as e:
            print(f"Error serializing results to JSON: {e}")
            print("Problematic history dict structure:")
            for k, v in serializable_history.items():
                 print(f"  {k}: {type(v)}")
            # Save what we can
            serializable_history['layer_stats_at_pruning'] = "Error during serialization"
            with open(results_path, 'w') as f:
                json.dump(serializable_history, f, indent=2)
                print(f"Saved partial results to {results_path} with serialization error.")

        # Save plots
        self.plot_results(results_dir)

        # Save model state
        model_save_path = os.path.join(results_dir, 'final_model.pth')
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Results and final model saved to {results_dir}")


    def plot_results(self, results_dir):
        epochs = range(len(self.history['train_loss']))

        plt.figure(figsize=(12, 8))

        # Loss Plot
        plt.subplot(2, 2, 1)
        plt.plot(epochs, self.history['train_loss'], label='Train Loss')
        plt.plot(epochs, self.history['test_loss'], label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss')
        plt.legend()
        plt.grid(True)

        # Accuracy Plot
        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.history['test_accuracy'], label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Test Accuracy')
        plt.legend()
        plt.grid(True)

        # Pruning Rate Plot
        plt.subplot(2, 2, 3)
        plt.plot(epochs, [p * 100 for p in self.history['pruning_rate']], label='Pruning Rate (%)')
        if self.history['pruning_applied_epoch'] != -1:
            plt.axvline(x=self.history['pruning_applied_epoch'], color='r', linestyle='--', label=f'Pruning Applied (Epoch {self.history["pruning_applied_epoch"]})')
        plt.xlabel('Epochs')
        plt.ylabel('Pruning Rate (%)')
        plt.title('Model Pruning Rate Over Training')
        plt.legend()
        plt.grid(True)

        # Epoch Times Plot
        plt.subplot(2, 2, 4)
        plt.plot(epochs, self.history['epoch_times'], label='Epoch Duration')
        plt.xlabel('Epochs')
        plt.ylabel('Time (s)')
        plt.title('Time Per Epoch')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(results_dir, 'training_plots.png')
        plt.savefig(plot_path)
        print(f"Training plots saved to {plot_path}")
        plt.close() 