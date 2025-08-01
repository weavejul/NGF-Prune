import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

class PrunableLinear(nn.Linear):
    """Linear layer with an output mask for neuron pruning."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Mask applied to output neurons
        self.register_buffer('mask', torch.ones(self.out_features, dtype=torch.float32))
        self.register_buffer('activation_sum', torch.zeros(self.out_features))
        self.register_buffer('activation_count', torch.zeros(1))
        self.pruned = False # Has pruning been applied?

    def forward(self, x):
        output = F.linear(x, self.weight, self.bias)
        # Apply mask to output neurons
        masked_output = output * self.mask.unsqueeze(0)
        return masked_output

    def reset_activation_stats(self):
        self.activation_sum.zero_()
        self.activation_count.zero_()

    def get_average_activation(self):
        if self.activation_count.item() == 0:
            return torch.zeros_like(self.activation_sum)
        return self.activation_sum / self.activation_count.item()

    def update_mask(self, new_mask):
        """Updates mask and marks layer as pruned"""
        if self.mask.shape != new_mask.shape:
             raise ValueError(f"New mask shape {new_mask.shape} does not match existing mask shape {self.mask.shape}")
        self.mask.data = new_mask.to(self.mask.device, dtype=self.mask.dtype)
        self.pruned = True
        # Zero out pruned neuron weights
        with torch.no_grad():
            self.weight.data *= self.mask.unsqueeze(1)
            if self.bias is not None:
                self.bias.data *= self.mask


class PrunableConv2d(nn.Conv2d):
    """Conv2d layer with an output channel mask for pruning."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Mask applied to output channels
        self.register_buffer('mask', torch.ones(self.out_channels, dtype=torch.float32))
        # Stats per output channel
        self.register_buffer('activation_sum', torch.zeros(self.out_channels))
        self.register_buffer('activation_count', torch.zeros(1))
        self.pruned = False

    def forward(self, x):
        output = super().forward(x)
        # Apply mask to output channels
        masked_output = output * self.mask.view(1, -1, 1, 1)
        return masked_output

    def reset_activation_stats(self):
        self.activation_sum.zero_()
        self.activation_count.zero_()

    def get_average_activation(self):
        if self.activation_count.item() == 0:
            return torch.zeros_like(self.activation_sum)
        # Avg activation magnitude per channel
        return self.activation_sum / self.activation_count.item()

    def update_mask(self, new_mask):
        """Updates the mask and marks the layer as pruned."""
        if self.mask.shape != new_mask.shape:
             raise ValueError(f"New mask shape {new_mask.shape} does not match existing mask shape {self.mask.shape}")
        self.mask.data = new_mask.to(self.mask.device, dtype=self.mask.dtype)
        self.pruned = True
        # Zero out weights and biases corresponding to pruned channels
        with torch.no_grad():
            self.weight.data *= self.mask.view(-1, 1, 1, 1) # Zero filters (output channels)
            if self.bias is not None:
                self.bias.data *= self.mask


# Activation Monitoring Hooks
def activation_hook(module, input, output):
    """Forward hook to record activation magnitudes."""
    if isinstance(module, (PrunableLinear, PrunableConv2d)):
        # Hook masked output (for L1 norm)
        with torch.no_grad():
            if isinstance(module, PrunableLinear):
                activation_mag = torch.abs(output).mean(dim=0)
            elif isinstance(module, PrunableConv2d):
                # Avg abs activation per output channel across batch, H, W
                activation_mag = torch.abs(output).mean(dim=[0, 2, 3])
            else:
                return

            module.activation_sum += activation_mag.to(module.activation_sum.device)
            module.activation_count += 1


def add_activation_hooks(model):
    """Adds forward hooks to Prunable layers in the model"""
    handles = []
    for layer in model.modules():
        if isinstance(layer, (PrunableLinear, PrunableConv2d)):
            handle = layer.register_forward_hook(activation_hook)
            handles.append(handle)
            # Reset stats before monitoring
            layer.reset_activation_stats()
            print(f"Added hook to layer: {layer.__class__.__name__}")
    return handles

def remove_hooks(handles):
    """Removes previously added hooks."""
    for handle in handles:
        handle.remove()



def calculate_ngf_masks(model, keep_fraction):
    """
    Calculates new masks for Prunable layers based on relative activation magnitude.
    Neurons/channels w/ avg activation below the keep_fraction percentile are pruned.

    Args:
        model (nn.Module): The model containing Prunable layers.
        keep_fraction (float): The fraction of neurons/channels to KEEP in each layer
                               (e.g., 0.7 keeps the top 70% most active).

    Returns:
        tuple: (dict_of_masks, float_overall_pruning_rate, dict_layer_stats)
               Returns (None, 0.0, {}) if no prunable layers found or stats issues.
    """
    if not (0.0 < keep_fraction <= 1.0):
        raise ValueError(f"keep_fraction must be between 0.0 (exclusive) and 1.0 (inclusive), got {keep_fraction}")

    new_masks = {}
    layer_stats = {} # Per-layer stats
    prunable_layers_found = False
    total_neurons = 0
    pruned_neurons = 0

    for name, module in model.named_modules():
        if isinstance(module, (PrunableLinear, PrunableConv2d)):
            prunable_layers_found = True
            layer_total = module.mask.numel()
            layer_pruned = 0
            layer_active = layer_total
            activation_threshold_value = 0.0 # Actual magnitude threshold used
            avg_act_range = [0.0, 0.0]
            current_mask = module.mask.data # Get current mask

            if module.activation_count.item() > 0:
                avg_activation = module.get_average_activation()
                avg_act_range = [avg_activation.min().item(), avg_activation.max().item()]

                if layer_total > 0:
                    # Determine percentile threshold value
                    prune_percentile = (1.0 - keep_fraction) * 100
                    activation_threshold_value = np.percentile(avg_activation.cpu().numpy(), prune_percentile)
                    # Create mask
                    calculated_mask = (avg_activation >= activation_threshold_value).float()
                    calculated_mask = calculated_mask.to(module.mask.device, dtype=module.mask.dtype)
                    new_masks[name] = calculated_mask

                    layer_active = calculated_mask.sum().item()
                    layer_pruned = layer_total - layer_active
                    print(f"Layer '{name}' (NGF): Keep Frac: {keep_fraction:.2f}, Activ Range: [{avg_act_range[0]:.4f}, {avg_act_range[1]:.4f}], Threshold Val: {activation_threshold_value:.4f}, Pruning {layer_pruned}/{layer_total} neurons ({layer_pruned/layer_total*100:.2f}%)")

            elif module.activation_count.item() == 0:
                # Handles case where stats weren't collected for some reason
                print(f"Warning: Layer '{name}' has no activation stats collected. Keeping mask as ones.")
                new_masks[name] = torch.ones_like(current_mask)
                layer_active = layer_total
                layer_pruned = 0

            # Store stats for current layer
            layer_stats[name] = {
                'total_neurons': layer_total,
                'pruned_neurons': layer_pruned,
                'active_neurons': layer_active,
                'pruning_rate': layer_pruned / layer_total if layer_total > 0 else 0.0,
                'avg_activation_min': avg_act_range[0],
                'avg_activation_max': avg_act_range[1],
                'activation_threshold_value': float(activation_threshold_value)
            }
            total_neurons += layer_total
            pruned_neurons += layer_pruned

    if not prunable_layers_found:
        print("No PrunableLinear or PrunableConv2d layers found in the model.")
        return None, 0.0, {}

    overall_pruning_rate = (pruned_neurons / total_neurons) if total_neurons > 0 else 0.0
    print(f"Overall NGF Pruning Calculation: Pruned {pruned_neurons}/{total_neurons} neurons ({overall_pruning_rate*100:.2f}%)")

    return new_masks, overall_pruning_rate, layer_stats


def apply_masks(model, masks_dict):
    """
    Applies the calculated masks to the corresponding Prunable layers in the model.

    Args:
        model (nn.Module): The model containing Prunable layers.
        masks_dict (dict): Dictionary mapping layer names to new mask tensors.
    """
    if masks_dict is None:
        print("No masks provided to apply.")
        return
    
    layers_updated = 0
    for name, module in model.named_modules():
        if name in masks_dict:
            if isinstance(module, (PrunableLinear, PrunableConv2d)):
                new_mask = masks_dict[name]
                if not torch.equal(module.mask.data, new_mask):
                    print(f"Applying updated mask to layer '{name}' ({int(new_mask.sum())}/{new_mask.numel()} active)")
                    module.update_mask(new_mask)
                    layers_updated += 1
                else:
                    pass
            else:
                 print(f"Warning: Layer '{name}' found in masks_dict but is not PrunableLinear or PrunableConv2d.")
    if layers_updated == 0:
        print("No layer masks were changed in this pruning step.")

def convert_to_prunable(model):
    """
    Recursively converts nn.Linear and nn.Conv2d layers in a model
    to PrunableLinear and PrunableConv2d layers, preserving weights.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            new_module = PrunableLinear(module.in_features, module.out_features, bias=module.bias is not None)
            new_module.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_module.bias.data = module.bias.data.clone()
            setattr(model, name, new_module.to(module.weight.device))
            print(f"Converted layer '{name}' to PrunableLinear")
        elif isinstance(module, nn.Conv2d):
            new_module = PrunableConv2d(
                module.in_channels, module.out_channels, module.kernel_size,
                module.stride, module.padding, module.dilation, module.groups,
                bias=module.bias is not None, padding_mode=module.padding_mode
            )
            new_module.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_module.bias.data = module.bias.data.clone()
            setattr(model, name, new_module.to(module.weight.device))
            print(f"Converted layer '{name}' to PrunableConv2d")
        else:
            convert_to_prunable(module)


def get_total_neurons(model):
    """Counts total neurons/channels in Prunable layers"""
    count = 0
    for module in model.modules():
        if isinstance(module, (PrunableLinear, PrunableConv2d)):
            count += module.mask.numel()
    return count

def get_active_neurons(model):
    """Counts active (unpruned) neurons/channels in Prunable layers"""
    count = 0
    for module in model.modules():
        if isinstance(module, (PrunableLinear, PrunableConv2d)):
            count += module.mask.sum().item()
    return count

def get_model_pruning_rate(model):
    """Calculates the overall pruning rate of the model"""
    total = get_total_neurons(model)
    if total == 0:
        return 0.0
    active = get_active_neurons(model)
    return (total - active) / total 

def calculate_weight_sparsity(model):
    """Calculates the overall weight sparsity of the model 
       (fraction of zero weights in Linear and Conv2d layers)"""
    total_weights = 0
    zero_weights = 0
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            total_weights += module.weight.nelement()
            zero_weights += torch.sum(module.weight == 0).item()
                
    if total_weights == 0:
        return 0.0
        
    sparsity = zero_weights / total_weights
    return sparsity 