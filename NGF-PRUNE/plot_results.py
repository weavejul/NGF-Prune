import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Import seaborn
import argparse
import numpy as np
from collections import defaultdict
import glob
import warnings

def load_results(results_dir, dataset_filter='fashion_mnist'):
    """Loads all experiment results from subdirectories."""
    all_data = []
    # Updated pattern for nested structure: results/{dataset}/{model}/{run_dir}/results.json
    # We search within the specific dataset directory provided by dataset_filter
    pattern = os.path.join(results_dir, dataset_filter, "*", "*", "results.json")
    result_files = glob.glob(pattern, recursive=False) # No need for recursive here

    print(f"Searching for results with pattern: '{pattern}'.")
    print(f"Found {len(result_files)} result files.")

    for result_file in result_files:
        exp_dir = os.path.dirname(result_file)
        config_file = os.path.join(exp_dir, "config.json")

        if not os.path.exists(config_file):
            # print(f"Warning: Config file not found in {exp_dir}, skipping.") # Reduce noise
            continue

        try:
            with open(result_file, 'r') as f:
                results = json.load(f)
            with open(config_file, 'r') as f:
                config = json.load(f)

            # Extract key information
            pruning_method = config.get('pruning_method')
            critical_start = config.get('critical_start_epoch')
            critical_duration = config.get('critical_duration')
            magnitude_apply_epoch = config.get('pruning_apply_epoch')
            pruning_threshold = config.get('pruning_threshold')

            # Determine the epoch pruning was actually applied
            pruning_applied_epoch = results.get('pruning_applied_epoch', -1)
            # If not found in results, infer from config (for older runs or fallback)
            if pruning_applied_epoch == -1:
                if pruning_method == 'ngf' or pruning_method == 'ngf_scheduled': # Handle scheduled too
                    pruning_applied_epoch = critical_start + critical_duration if critical_start is not None and critical_duration is not None else None
                elif pruning_method == 'magnitude':
                    pruning_applied_epoch = magnitude_apply_epoch
                else:
                    pruning_applied_epoch = None

            # --- Extract Epoch-wise data --- 
            train_loss_list = results.get('train_loss', [])
            test_loss_list = results.get('test_loss', [])
            eval_acc_list = results.get('eval_acc', []) # Use 'eval_acc' if present
            test_acc_list = results.get('test_accuracy', []) # Fallback
            epochs_completed_list = list(range(len(train_loss_list)))

            # Check list lengths
            valid_epoch_data = True
            if len(train_loss_list) != len(test_loss_list) or not test_loss_list:
                print(f"Warning: Train/Test loss list length mismatch or empty in {result_file}. Min loss set to NaN.")
                min_loss = np.nan
                train_loss_list, test_loss_list, epochs_completed_list = [], [], []
                valid_epoch_data = False
            else:
                 min_loss = min(test_loss_list) # Calculate min test loss for this run

            # Determine which accuracy list to use
            accuracy_list_to_use = eval_acc_list if eval_acc_list else test_acc_list
            epochs_run_count = len(epochs_completed_list)

            if len(accuracy_list_to_use) != epochs_run_count:
                 print(f"Warning: Accuracy list length ({len(accuracy_list_to_use)}) mismatch with train loss length ({epochs_run_count}) in {result_file}.")
                 # Optionally truncate or pad, for now just warn
            
            # --- Initial Result Values --- 
            loaded_final_pruning_rate = results.get('final_pruning_rate')
            layer_stats = results.get('layer_stats_at_pruning', {})

            # --- RECALCULATION LOGIC FOR OLD MAGNITUDE RUNS --- 
            recalculated_rate = None
            if pruning_method == 'magnitude' and loaded_final_pruning_rate is not None and np.isclose(loaded_final_pruning_rate, 0.0):
                if isinstance(layer_stats, dict) and layer_stats:
                    total_p = 0
                    pruned_p = 0
                    valid_stats = True
                    for layer_name, stats in layer_stats.items():
                        if isinstance(stats, dict) and 'total_params' in stats and 'pruned_params' in stats:
                            total_p += stats['total_params']
                            pruned_p += stats['pruned_params']
                        else:
                            valid_stats = False
                            break
                    
                    if valid_stats and total_p > 0:
                        recalculated_rate = pruned_p / total_p
                        print(f"Info: Recalculated Magnitude sparsity for {exp_dir}. Original: {loaded_final_pruning_rate:.4f}, Recalculated: {recalculated_rate:.4f}")
                    elif not valid_stats:
                         print(f"Warning: Invalid layer_stats format found in {exp_dir} for Magnitude run. Cannot recalculate sparsity.")
                    else: # total_p was 0
                         print(f"Warning: Layer stats in {exp_dir} reported 0 total parameters for Magnitude run. Sparsity is 0.")
                else:
                     print(f"Warning: Magnitude run {exp_dir} has 0 sparsity but no valid layer_stats to recalculate from.")
            # --- END RECALCULATION LOGIC --- 
            
            final_pruning_rate_to_use = recalculated_rate if recalculated_rate is not None else loaded_final_pruning_rate

            data_point = {
                # --- Config data --- 
                'model': config.get('model'),
                'dataset': config.get('dataset'),
                'seed': config.get('seed'),
                'pruning_method': pruning_method,
                'ngf_start_epoch': critical_start if pruning_method in ['ngf', 'ngf_scheduled'] else None,
                'ngf_duration': critical_duration if pruning_method in ['ngf', 'ngf_scheduled'] else None,
                'mag_apply_epoch': magnitude_apply_epoch if pruning_method == 'magnitude' else None,
                'pruning_threshold_config': pruning_threshold,
                'lr': config.get('lr'),
                'ngf_keep_fraction': pruning_threshold if pruning_method in ['ngf', 'ngf_scheduled'] else None,
                'mag_prune_fraction': pruning_threshold if pruning_method == 'magnitude' else None,

                # --- Summary Results --- 
                'best_accuracy': results.get('best_accuracy'),
                'final_accuracy': results.get('final_accuracy'),
                'final_pruning_rate': final_pruning_rate_to_use,
                'training_time': results.get('total_training_time'),
                'epochs_run': epochs_run_count,
                'pruning_applied_epoch': pruning_applied_epoch,
                'layer_stats': layer_stats,
                'min_test_loss': min_loss,
                'results_dir': exp_dir,

                # --- Epoch-wise Lists (optional for memory) --- 
                'epoch': epochs_completed_list if valid_epoch_data else [],
                'train_loss': train_loss_list if valid_epoch_data else [],
                'test_loss': test_loss_list if valid_epoch_data else [],
                'eval_acc': eval_acc_list if valid_epoch_data else [],
                'test_accuracy': test_acc_list if valid_epoch_data else []
            }

            # Handle potential None values for aggregation
            numeric_cols = ['best_accuracy', 'final_accuracy', 'final_pruning_rate', 'training_time', 'min_test_loss']
            for key in numeric_cols:
                if data_point[key] is None:
                    # print(f"Warning: Missing value for '{key}' in {result_file}") # Reduce noise
                    data_point[key] = np.nan
                elif not isinstance(data_point[key], (int, float)):
                     try:
                         data_point[key] = float(data_point[key])
                     except (ValueError, TypeError):
                         print(f"Warning: Could not convert '{key}' value '{data_point[key]}' to float in {result_file}. Setting to NaN.")
                         data_point[key] = np.nan

            # No need to refine threshold meaning here, just store raw lists
            # Keep the code that sets ngf_keep_fraction / mag_prune_fraction for filtering
            if data_point['pruning_method'] == 'ngf':
                pass # Already set above
            elif data_point['pruning_method'] == 'magnitude':
                pass # Already set above
            else: # 'none'
                pass # Already set above

            all_data.append(data_point)

        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in {result_file}, skipping.")
        except Exception as e:
            print(f"Error loading results from {exp_dir}: {e}")

    if not all_data:
        print(f"No valid results found in {results_dir} for dataset {dataset_filter}. Exiting.")
        return pd.DataFrame()

    return pd.DataFrame(all_data)

def aggregate_results(df):
    """Aggregates results by configuration, calculating mean and std dev."""
    if df.empty:
        return df
        
    df_clean = df.dropna(subset=['model', 'pruning_method'])
    if df_clean.empty:
        return df_clean

    agg_funcs = {
        'best_accuracy': ['mean', 'std'],
        'final_accuracy': ['mean', 'std'],
        'final_pruning_rate': ['mean', 'std'],
        'training_time': ['mean', 'std'],
        'min_test_loss': ['mean', 'std'],
        'epochs_run': ['mean'],
        'seed': 'count'
    }

    aggregated_list = []
    try:
        # Ensure correct handling of potential missing columns during grouping
        for method, method_df in df_clean.groupby('pruning_method', dropna=False):
            grouping_cols = ['model', 'pruning_method']
            potential_cols = []
            if method == 'ngf' or method == 'ngf_scheduled':
                potential_cols = ['pruning_applied_epoch', 'ngf_keep_fraction', 'ngf_duration', 'ngf_start_epoch']
            elif method == 'magnitude':
                potential_cols = ['pruning_applied_epoch', 'mag_prune_fraction']
            
            valid_method_cols = [col for col in potential_cols if col in method_df.columns and method_df[col].notna().any()]
            grouping_cols.extend(valid_method_cols)
            
            # Check if all grouping columns exist
            final_grouping_cols = [col for col in grouping_cols if col in method_df.columns]
            if not final_grouping_cols:
                print(f"Warning: No valid grouping columns found for method {method}. Skipping.")
                continue
                
            # Group and aggregate, handling potential empty groups
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning) # Ignore mean of empty slice
                method_agg = method_df.groupby(final_grouping_cols, dropna=False).agg(agg_funcs)
            
            if method_agg.empty:
                continue

            method_agg.columns = ['_'.join(map(str, col)).strip('_') for col in method_agg.columns.values]
            method_agg = method_agg.rename(columns={'seed_count': 'run_count'})
            method_agg = method_agg.reset_index()
            aggregated_list.append(method_agg)

        if not aggregated_list:
             print("Warning: No data found after grouping and aggregation.")
             return pd.DataFrame()

        # Concatenate, ensuring consistent columns
        aggregated = pd.concat(aggregated_list, ignore_index=True, sort=False)

    except KeyError as e:
        print(f"KeyError during grouping/aggregation: {e}. Columns in df_clean: {df_clean.columns}")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred during aggregation: {e}")
        raise e

    return aggregated

def set_plot_style():
    """Sets a visually appealing default style for plots."""
    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.1)
    plt.rcParams['figure.figsize'] = (12, 7)
    plt.rcParams['figure.titlesize'] = 18
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['legend.title_fontsize'] = 12
    plt.rcParams['errorbar.capsize'] = 3
    plt.rcParams['lines.markersize'] = 7
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.6

def get_baseline_metrics(df_agg, model_name):
    """Extracts baseline metrics for a given model."""
    baseline_df = df_agg[(df_agg['model'] == model_name) & (df_agg['pruning_method'] == 'none')]
    if not baseline_df.empty:
        metrics = baseline_df.iloc[0]
        # Ensure we return NaN if baseline accuracy itself is NaN
        baseline_acc_mean = metrics.get('best_accuracy_mean', np.nan)
        return {
            'accuracy_mean': baseline_acc_mean,
            'accuracy_std': metrics.get('best_accuracy_std', np.nan) if not np.isnan(baseline_acc_mean) else np.nan,
            'time_mean': metrics.get('training_time_mean', np.nan),
            'time_std': metrics.get('training_time_std', np.nan),
            'min_loss_mean': metrics.get('min_test_loss_mean', np.nan),
            'min_loss_std': metrics.get('min_test_loss_std', np.nan)
        }
    else:
        print(f"Warning: Baseline data not found for model '{model_name}'")
        return None

def plot_accuracy_drop_vs_sparsity(df_agg, output_dir, model_name, dataset_name, baseline_metrics):
    """Plots accuracy drop vs. sparsity for NGF methods ONLY."""
    set_plot_style()
    
    if baseline_metrics is None or np.isnan(baseline_metrics['accuracy_mean']):
        print(f"Skipping Accuracy Drop plot for {model_name}: Baseline accuracy not available.")
        return

    baseline_acc = baseline_metrics['accuracy_mean']
    
    # Filter data: Select only NGF methods and calculate accuracy drop/sparsity
    plot_data = df_agg[(df_agg['model'] == model_name) & 
                       (df_agg['pruning_method'].isin(['ngf', 'ngf_scheduled']))].copy()
    
    if plot_data.empty:
        print(f"No NGF data found for {model_name} to plot accuracy drop.")
        return
        
    plot_data['accuracy_drop'] = baseline_acc - plot_data['best_accuracy_mean']
    plot_data['sparsity_pct_mean'] = plot_data['final_pruning_rate_mean'] * 100
    # Method is now implicitly NGF
    # plot_data['Method'] = plot_data['pruning_method'].apply(lambda x: 'NGF' if x in ['ngf', 'ngf_scheduled'] else 'Magnitude')
    
    # --- Plot (Single method: NGF) ---
    plt.figure(figsize=(10, 8))
    # Removed hue and style as they are constant (NGF)
    # Add other NGF params to style/size/hue if desired (e.g., hue=duration, size=keep_fraction)
    sns.scatterplot(data=plot_data, x='sparsity_pct_mean', y='accuracy_drop',
                    hue='ngf_duration', # Example: Color by duration
                    size='ngf_keep_fraction', # Example: Size by keep fraction
                    style='ngf_start_epoch', # Example: Style by start epoch
                    sizes=(50, 300),
                    s=150, alpha=0.8, legend='auto') # Removed style='Method'
    
    # Add error bars for accuracy drop
    plt.errorbar(plot_data['sparsity_pct_mean'], plot_data['accuracy_drop'], 
                 yerr=plot_data['best_accuracy_std'], fmt='none', color='grey', alpha=0.5, zorder=-1)

    # Update title and labels for NGF only
    plt.title(f'{model_name.upper()} NGF Accuracy Drop vs. Neuron/Channel Sparsity ({dataset_name})')
    plt.xlabel('Neuron/Channel Sparsity (% Pruned)') 
    plt.ylabel('Accuracy Drop (Baseline Acc - Pruned Acc)')
    plt.axhline(0, color='black', linestyle='--', linewidth=1, label='No Accuracy Drop')
    plt.grid(True)
    plt.legend(title='NGF Parameters', bbox_to_anchor=(1.05, 1), loc='upper left') # Updated legend title
    
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for external legend
    save_path = os.path.join(output_dir, f'{model_name}_ACC_DROP_vs_SPARSITY.png')
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    plt.close()

def plot_layerwise_sparsity(df_raw, output_dir, model_name, dataset_name):
    """Plots the average sparsity per layer for different pruning methods."""
    set_plot_style()

    layer_data = []
    model_df = df_raw[(df_raw['model'] == model_name) & (df_raw['layer_stats'].apply(lambda x: isinstance(x, dict) and bool(x)))].copy() # Ensure layer_stats is a non-empty dict
    
    if model_df.empty:
        print(f"Skipping Layerwise Sparsity plot for {model_name}: No valid 'layer_stats' data found.")
        return

    for _, row in model_df.iterrows():
        method = row['pruning_method']
        if method not in ['ngf', 'ngf_scheduled', 'magnitude']:
            continue
            
        stats = row['layer_stats']
        if not isinstance(stats, dict): continue # Skip if not a dict
            
        method_label = 'NGF' if method in ['ngf', 'ngf_scheduled'] else 'Magnitude'

        for layer_name, layer_info in stats.items():
            if isinstance(layer_info, dict) and 'pruning_rate' in layer_info:
                pruning_rate = layer_info['pruning_rate']
                if pruning_rate is not None: # Check for None
                    sparsity = pruning_rate * 100 # Use the stored pruning rate
                    layer_data.append({
                        'Method': method_label,
                        'Layer': layer_name,
                        'Sparsity': sparsity,
                        'run_id': row.get('results_dir', 'unknown') # For debugging if needed
                    })
             # else: # Handle cases where layer_info might not be a dict or missing key
             #    print(f"Warning: Invalid layer_info for layer '{layer_name}' in run {row.get('results_dir')}: {layer_info}")


    if not layer_data:
        print(f"No processable layer sparsity data found for {model_name} after parsing.")
        return

    layer_df = pd.DataFrame(layer_data)
    
    # --- Plot ---
    plt.figure(figsize=(14, 8))
    # Calculate mean sparsity per layer per method
    agg_layer_df = layer_df.groupby(['Method', 'Layer'])['Sparsity'].mean().reset_index()
    
    # Ensure consistent layer order if possible (e.g., sort alphanumeric)
    layer_order = sorted(layer_df['Layer'].unique()) 

    # Pass raw data and let barplot compute mean and std dev error bars
    sns.barplot(data=layer_df, x='Layer', y='Sparsity', hue='Method', 
                order=layer_order, palette='viridis', errorbar='sd')

    plt.title(f'{model_name.upper()} Average Layer Sparsity ({dataset_name})\n(NGF: Neuron/Channel, Magnitude: Weight - Based on Recorded Pruning Rate)')
    plt.xlabel('Model Layer')
    plt.ylabel('Average Sparsity (% Pruned Units)')
    plt.xticks(rotation=45, ha='right') # Rotate labels if many layers
    plt.grid(axis='y')
    plt.legend(title='Pruning Method')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{model_name}_LAYERWISE_SPARSITY.png')
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    plt.close()

def plot_accuracy_time_frontier(df_agg, output_dir, model_name, dataset_name, baseline_metrics):
    """Plots Best Accuracy vs. Training Time trade-off (Pareto-like)."""
    set_plot_style()

    plot_data = df_agg[(df_agg['model'] == model_name) &
                       (df_agg['pruning_method'].isin(['none', 'ngf', 'ngf_scheduled', 'magnitude']))].copy()

    if plot_data.empty:
        print(f"No data found for {model_name} to plot accuracy-time frontier.")
        return

    # Simplify method names for plotting
    def simplify_method_frontier(method):
        if method in ['ngf', 'ngf_scheduled']:
            return 'NGF'
        elif method == 'magnitude':
            return 'Magnitude'
        else: # 'none'
            return 'Baseline'
    plot_data['Method'] = plot_data['pruning_method'].apply(simplify_method_frontier)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=plot_data, x='training_time_mean', y='best_accuracy_mean',
                    hue='Method', style='Method', s=150, alpha=0.8,
                    palette=sns.color_palette("colorblind", n_colors=plot_data['Method'].nunique())) # Ensure distinct colors

    # Add error bars
    # Plot error bars separately for each method to avoid connecting unrelated points
    methods = plot_data['Method'].unique()
    colors = sns.color_palette("colorblind", n_colors=len(methods))
    method_colors = dict(zip(methods, colors))

    for method in methods:
        method_df = plot_data[plot_data['Method'] == method]
        plt.errorbar(method_df['training_time_mean'], method_df['best_accuracy_mean'],
                     xerr=method_df['training_time_std'], yerr=method_df['best_accuracy_std'],
                     fmt='none', color=method_colors[method], alpha=0.3, zorder=-1)

    plt.title(f'{model_name.upper()} Accuracy vs. Training Time ({dataset_name})')
    plt.xlabel('Average Training Time (seconds)')
    plt.ylabel('Average Best Test Accuracy (%)')
    plt.grid(True)
    plt.legend(title='Pruning Method')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{model_name}_ACC_vs_TIME_Frontier.png')
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    plt.close()

def plot_loss_curves(df_raw, output_dir, model_name, dataset_name):
    """Plots average training and testing loss curves over epochs."""
    set_plot_style()

    # Filter raw data for the model, requiring epoch data
    model_df = df_raw[(df_raw['model'] == model_name) &
                      (df_raw['train_loss'].apply(lambda x: isinstance(x, list) and len(x) > 0)) &
                      (df_raw['test_loss'].apply(lambda x: isinstance(x, list) and len(x) > 0))].copy()

    if model_df.empty:
        print(f"Skipping Loss Curve plot for {model_name}: No valid epoch data found.")
        return

    # Unpack epoch data into a long-form DataFrame
    loss_data = []
    max_epochs = 0
    for _, row in model_df.iterrows():
        method = row['pruning_method']
        method_label = 'NGF' if method in ['ngf', 'ngf_scheduled'] else ('Magnitude' if method == 'magnitude' else 'Baseline')
        run_id = row.get('results_dir', 'unknown') # Unique identifier for the run
        epochs = row['epoch']
        train_losses = row['train_loss']
        test_losses = row['test_loss']
        
        # Ensure lists have same length as epochs list
        min_len = min(len(epochs), len(train_losses), len(test_losses))
        if min_len != len(epochs):
            print(f"Warning: Loss list length mismatch for run {run_id}. Truncating to {min_len} epochs.")

        for i in range(min_len):
            loss_data.append({'Method': method_label, 'Epoch': epochs[i], 'Loss Type': 'Train', 'Loss': train_losses[i], 'Run': run_id})
            loss_data.append({'Method': method_label, 'Epoch': epochs[i], 'Loss Type': 'Test', 'Loss': test_losses[i], 'Run': run_id})
        if min_len > max_epochs:
            max_epochs = min_len

    if not loss_data:
        print(f"No processable loss data found for {model_name}.")
        return

    loss_df = pd.DataFrame(loss_data)

    # --- Plot --- 
    # Use relplot for separate Train/Test facets, using lineplot internally
    # ci='sd' shows standard deviation across runs
    g = sns.relplot(data=loss_df, x='Epoch', y='Loss', hue='Method', col='Loss Type', 
                    kind='line', errorbar='sd', facet_kws={'sharey': False}, # Don't share y-axis for Train/Test
                    height=6, aspect=1.2, 
                    palette=sns.color_palette("colorblind", n_colors=loss_df['Method'].nunique()))

    g.fig.suptitle(f'{model_name.upper()} Average Loss Curves ({dataset_name})', y=1.03) # Adjust title position
    g.set_axis_labels('Epoch', 'Average Loss')
    g.set_titles("{col_name} Loss")
    
    # Add grid lines manually if desired, as relplot doesn't add them automatically to facets
    for ax in g.axes.flat:
        ax.grid(True, linestyle='--', alpha=0.6)
        # Optional: Limit x-axis if needed
        # ax.set_xlim(0, max_epochs) 

    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout slightly for title
    save_path = os.path.join(output_dir, f'{model_name}_LOSS_CURVES.png')
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    plt.close()

def plot_accuracy_distribution(df_raw, output_dir, model_name, dataset_name):
    """Plots the distribution of best accuracy for each method using violin plots."""
    set_plot_style()

    # Filter raw data for the model, needing best_accuracy
    model_df = df_raw[(df_raw['model'] == model_name) &
                      (df_raw['best_accuracy'].notna())].copy()

    if model_df.empty:
        print(f"Skipping Accuracy Distribution plot for {model_name}: No valid best_accuracy data found.")
        return

    # Simplify method names
    def simplify_method(row):
        if row['pruning_method'] in ['ngf', 'ngf_scheduled']:
            return 'NGF'
        elif row['pruning_method'] == 'magnitude':
            return 'Magnitude'
        else: # 'none'
            return 'Baseline'
    model_df['Method'] = model_df.apply(simplify_method, axis=1)

    # --- Plot --- 
    plt.figure(figsize=(10, 7))
    # Address palette warning by assigning hue explicitly
    sns.violinplot(data=model_df, x='Method', y='best_accuracy', hue='Method', 
                   palette="colorblind", inner='quartile', legend=False)
    sns.stripplot(data=model_df, x='Method', y='best_accuracy', 
                  color='.3', alpha=0.5, jitter=True)

    plt.title(f'{model_name.upper()} Distribution of Best Test Accuracy ({dataset_name})')
    plt.xlabel('Pruning Method')
    plt.ylabel('Best Test Accuracy (%)')
    plt.grid(True, axis='y') # Grid on y-axis only

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{model_name}_ACCURACY_DISTRIBUTION.png')
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    plt.close()

def plot_ngf_sensitivity(df_agg, output_dir, model_name, dataset_name):
    """Plots NGF performance (Accuracy vs Sparsity) vs. its hyperparameters."""
    set_plot_style()

    # Filter aggregated data specifically for NGF runs with necessary columns
    ngf_data = df_agg[(df_agg['model'] == model_name) &
                      (df_agg['pruning_method'].isin(['ngf', 'ngf_scheduled'])) &
                      (df_agg['ngf_keep_fraction'].notna()) &
                      (df_agg['ngf_duration'].notna()) &
                      (df_agg['ngf_start_epoch'].notna())].copy()

    if ngf_data.empty:
        print(f"Skipping NGF Sensitivity plot for {model_name}: No sufficient NGF data with varied hyperparameters found.")
        return

    # Calculate sparsity percentage
    ngf_data['sparsity_pct_mean'] = (1 - ngf_data['final_pruning_rate_mean']) * 100 # Note: using 1-rate for sparsity % 
    # Correction: final_pruning_rate IS the sparsity rate (0 to 1). Revert the above. 
    ngf_data['sparsity_pct_mean'] = ngf_data['final_pruning_rate_mean'] * 100

    # --- Plot --- 
    # Use relplot to potentially facet by one parameter if many runs exist
    # Or just a single scatterplot colored/styled by parameters
    plt.figure(figsize=(12, 8))
    # Example: size by keep_fraction, hue by duration, style by start_epoch
    sns.scatterplot(data=ngf_data, x='sparsity_pct_mean', y='best_accuracy_mean',
                    hue='ngf_duration', 
                    size='ngf_keep_fraction', 
                    style='ngf_start_epoch',
                    sizes=(50, 300), # Control range of point sizes
                    palette='viridis', alpha=0.8)

    # Add error bars (optional, can make plot busy)
    # plt.errorbar(ngf_data['sparsity_pct_mean'], ngf_data['best_accuracy_mean'],
    #              xerr=ngf_data['final_pruning_rate_std'] * 100, # x-error if needed
    #              yerr=ngf_data['best_accuracy_std'], 
    #              fmt='none', color='grey', alpha=0.3, zorder=-1)

    plt.title(f'{model_name.upper()} NGF Accuracy vs. Neuron/Channel Sparsity ({dataset_name})\nSensitivity to Hyperparameters')
    plt.xlabel('Average Neuron/Channel Sparsity (% Pruned)')
    plt.ylabel('Average Best Test Accuracy (%)')
    plt.grid(True)
    # Adjust legend position
    plt.legend(title='NGF Hyperparameters', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for external legend
    save_path = os.path.join(output_dir, f'{model_name}_NGF_SENSITIVITY.png')
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    plt.close()

def plot_sparsity_for_target_accuracy(df_raw, df_agg, output_dir, model_name, dataset_name, baseline_metrics, accuracy_tolerance=0.01):
    """Compares NGF Neuron/Channel Sparsity and Magnitude Weight Sparsity
       for runs achieving a similar target accuracy (e.g., near baseline)."""
    set_plot_style()

    if baseline_metrics is None or np.isnan(baseline_metrics['accuracy_mean']):
        print(f"Skipping Sparsity Comparison plot for {model_name}: Baseline accuracy not available.")
        return

    target_accuracy = baseline_metrics['accuracy_mean']
    acc_lower_bound = target_accuracy * (1 - accuracy_tolerance)
    acc_upper_bound = target_accuracy * (1 + accuracy_tolerance)

    # Filter RAW data for runs within the accuracy tolerance
    # Use raw data to get individual run sparsity values
    relevant_runs = df_raw[(
        (df_raw['model'] == model_name) &
        (df_raw['pruning_method'].isin(['ngf', 'ngf_scheduled', 'magnitude'])) &
        (df_raw['best_accuracy'].notna()) &
        (df_raw['final_pruning_rate'].notna()) &
        (df_raw['best_accuracy'] >= acc_lower_bound) &
        (df_raw['best_accuracy'] <= acc_upper_bound)
    )].copy()

    if relevant_runs.empty:
        print(f"Skipping Sparsity Comparison plot for {model_name}: No NGF or Magnitude runs found within tolerance ({accuracy_tolerance*100:.1f}%) of baseline accuracy ({target_accuracy:.2f}%).")
        return

    # Prepare data for plotting: Assign sparsity type and value
    plot_data = []
    for _, row in relevant_runs.iterrows():
        sparsity_value = row['final_pruning_rate'] * 100 # Percentage
        if row['pruning_method'] in ['ngf', 'ngf_scheduled']:
            plot_data.append({'Sparsity Type': 'Neuron/Channel (NGF)', 'Sparsity (%)': sparsity_value, 'Best Accuracy': row['best_accuracy']})
        elif row['pruning_method'] == 'magnitude':
            plot_data.append({'Sparsity Type': 'Weight (Magnitude)', 'Sparsity (%)': sparsity_value, 'Best Accuracy': row['best_accuracy']})

    if not plot_data:
        print(f"No data prepared for Sparsity Comparison plot for {model_name}.") # Should not happen if relevant_runs wasn't empty
        return

    plot_df = pd.DataFrame(plot_data)

    # --- Plot (Using Box Plot for Distribution Comparison) ---
    plt.figure(figsize=(10, 7))
    # Address palette warning by assigning hue explicitly
    sns.boxplot(data=plot_df, x='Sparsity Type', y='Sparsity (%)', hue='Sparsity Type',
                palette=["skyblue", "lightcoral"], legend=False)
    sns.stripplot(data=plot_df, x='Sparsity Type', y='Sparsity (%)',
                  color='.3', alpha=0.6, jitter=True)

    plt.title(f'{model_name.upper()} Sparsity Achieved at Near-Baseline Accuracy ({dataset_name})\n(Accuracy within +/- {accuracy_tolerance*100:.1f}% of Baseline: {target_accuracy:.2f}%)')
    plt.xlabel('Pruning Method and Sparsity Type')
    plt.ylabel('Sparsity (% Pruned)')
    plt.grid(True, axis='y')

    # Add text about how many runs are included
    n_ngf = len(plot_df[plot_df['Sparsity Type'] == 'Neuron/Channel (NGF)'])
    n_mag = len(plot_df[plot_df['Sparsity Type'] == 'Weight (Magnitude)'])
    plt.text(0.95, 0.01, f'NGF runs: {n_ngf}\nMagnitude runs: {n_mag}', 
             horizontalalignment='right', verticalalignment='bottom', 
             transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))


    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{model_name}_SPARSITY_COMPARISON_at_ACC.png')
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    plt.close()

def plot_loss_vs_time(df_raw, output_dir, model_name, dataset_name):
    """Plots average training and testing loss curves over estimated wall-clock time."""
    set_plot_style()

    # Filter raw data: need epoch lists, total time, and epochs run
    model_df = df_raw[(
        (df_raw['model'] == model_name) &
        (df_raw['train_loss'].apply(lambda x: isinstance(x, list) and len(x) > 0)) &
        (df_raw['test_loss'].apply(lambda x: isinstance(x, list) and len(x) > 0)) &
        (df_raw['training_time'].notna()) &
        (df_raw['training_time'] > 0) & # Avoid division by zero
        (df_raw['epochs_run'].notna()) &
        (df_raw['epochs_run'] > 0)
    )].copy()

    if model_df.empty:
        print(f"Skipping Loss vs. Time plot for {model_name}: No valid data with epoch lists, training_time, and epochs_run found.")
        return

    # Unpack data and estimate time per epoch
    loss_time_data = []
    max_time = 0
    for _, row in model_df.iterrows():
        method = row['pruning_method']
        method_label = 'NGF' if method in ['ngf', 'ngf_scheduled'] else ('Magnitude' if method == 'magnitude' else 'Baseline')
        run_id = row.get('results_dir', 'unknown')
        epochs = row['epoch']
        train_losses = row['train_loss']
        test_losses = row['test_loss']
        total_time = row['training_time']
        epochs_completed_count = row['epochs_run'] # Use the scalar value

        # Basic check
        if epochs_completed_count <= 0 or total_time <=0:
            print(f"Warning: Invalid epochs_run ({epochs_completed_count}) or training_time ({total_time}) for run {run_id}. Skipping run.")
            continue
            
        time_per_epoch = total_time / epochs_completed_count

        # Ensure list lengths match epochs list length reported
        min_len = min(len(epochs), len(train_losses), len(test_losses))
        if min_len == 0:
             print(f"Warning: Empty loss lists for run {run_id}. Skipping run.")
             continue
             
        # Use min_len which corresponds to actual data points we have
        actual_epochs_in_lists = min_len 
        # If list length is less than epochs_run, time_per_epoch calculation might be less accurate,
        # but it's the best estimate we have. 
        if actual_epochs_in_lists != epochs_completed_count:
             print(f"Warning: Loss list length ({actual_epochs_in_lists}) differs from epochs_run ({epochs_completed_count}) for run {run_id}. Using list length for plotting.")
             # Recalculate time_per_epoch based on actual data points? Or keep original? 
             # Let's keep original time_per_epoch derived from total_time and epochs_run for now.
             # time_per_epoch = total_time / actual_epochs_in_lists if actual_epochs_in_lists > 0 else 0 
        
        current_max_run_time = 0
        for i in range(actual_epochs_in_lists):
            # Estimate time at the end of this epoch
            time_at_epoch = time_per_epoch * (epochs[i] + 1)
            loss_time_data.append({'Method': method_label, 'Time (s)': time_at_epoch, 'Loss Type': 'Train', 'Loss': train_losses[i], 'Run': run_id})
            loss_time_data.append({'Method': method_label, 'Time (s)': time_at_epoch, 'Loss Type': 'Test', 'Loss': test_losses[i], 'Run': run_id})
            current_max_run_time = time_at_epoch

        if current_max_run_time > max_time:
            max_time = current_max_run_time

    if not loss_time_data:
        print(f"No processable loss vs time data found for {model_name}.")
        return

    loss_time_df = pd.DataFrame(loss_time_data)

    # --- Plot ---
    g = sns.relplot(data=loss_time_df, x='Time (s)', y='Loss', hue='Method', col='Loss Type',
                    kind='line', errorbar='sd', facet_kws={'sharey': False},
                    height=6, aspect=1.2,
                    palette=sns.color_palette("colorblind", n_colors=loss_time_df['Method'].nunique()))

    g.fig.suptitle(f'{model_name.upper()} Average Loss Curves vs. Training Time ({dataset_name})\n(Time estimated assuming constant epoch duration)', y=1.05) # Adjust title position & add note
    g.set_axis_labels('Estimated Training Time (seconds)', 'Average Loss')
    g.set_titles("{col_name} Loss")

    for ax in g.axes.flat:
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(0, None) # Start time at 0

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout slightly for title
    save_path = os.path.join(output_dir, f'{model_name}_LOSS_vs_TIME.png')
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    plt.close()

def plot_final_accuracy_vs_sparsity(df_agg, output_dir, model_name, dataset_name):
    """Plots Final Accuracy vs. Final Sparsity, distinguishing sparsity types."""
    set_plot_style()

    # Prepare dataframes, calculate sparsity percentages
    ngf_data = df_agg[(
        (df_agg['model'] == model_name) &
        (df_agg['pruning_method'].isin(['ngf', 'ngf_scheduled'])) &
        (df_agg['final_accuracy_mean'].notna()) &
        (df_agg['final_pruning_rate_mean'].notna())
    )].copy()
    if not ngf_data.empty:
        ngf_data['sparsity_pct_mean'] = ngf_data['final_pruning_rate_mean'] * 100

    mag_data = df_agg[(
        (df_agg['model'] == model_name) &
        (df_agg['pruning_method'] == 'magnitude') &
        (df_agg['final_accuracy_mean'].notna()) &
        (df_agg['final_pruning_rate_mean'].notna())
    )].copy()
    if not mag_data.empty:
         mag_data['sparsity_pct_mean'] = mag_data['final_pruning_rate_mean'] * 100

    if ngf_data.empty and mag_data.empty:
        print(f"Skipping Final Accuracy vs Sparsity plot for {model_name}: No valid NGF or Magnitude data found.")
        return
        
    # Get baseline final accuracy if available (using df_agg directly)
    baseline_final_acc = np.nan
    baseline_df = df_agg[(df_agg['model'] == model_name) & (df_agg['pruning_method'] == 'none')]
    if not baseline_df.empty:
        baseline_final_acc = baseline_df.iloc[0].get('final_accuracy_mean', np.nan) # Assuming aggregation includes final_accuracy stats
        # We might need to add final_accuracy to aggregate_results if not already there.
        # Let's check aggregate_results...
        # It's not there by default. Let's add it temporarily here for this plot if needed.
        # For now, let's assume final_accuracy is directly in df_agg from a potential custom aggregation, or we plot without baseline line.
        pass # We'll plot without baseline if final_accuracy_mean is not in df_agg.

    # --- Plot --- 
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    fig.suptitle(f'{model_name.upper()} Final Accuracy vs. Final Sparsity ({dataset_name})', fontsize=20)
    
    # NGF Panel
    if not ngf_data.empty:
        sns.scatterplot(data=ngf_data, x='sparsity_pct_mean', y='final_accuracy_mean', 
                      size='ngf_keep_fraction', hue='ngf_duration', style='ngf_start_epoch',
                      ax=axes[0], legend='auto', s=150, alpha=0.8)
        axes[0].errorbar(ngf_data['sparsity_pct_mean'], ngf_data['final_accuracy_mean'], 
                         yerr=ngf_data['final_accuracy_std'], fmt='none', color='grey', alpha=0.5, zorder=-1)
        axes[0].set_title('NGF Pruning')
        axes[0].set_xlabel('Neuron/Channel Sparsity (% Pruned)') 
        axes[0].set_ylabel('Average Final Test Accuracy (%)')
        axes[0].grid(True)
        if not np.isnan(baseline_final_acc):
             axes[0].axhline(baseline_final_acc, color='black', linestyle='--', label='Baseline Final Acc')
        axes[0].legend(title='NGF Params', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        axes[0].set_title('NGF Pruning (No Data)')
        axes[0].text(0.5, 0.5, 'No NGF Data Found', horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)

    # Magnitude Panel
    if not mag_data.empty:
        sns.scatterplot(data=mag_data, x='sparsity_pct_mean', y='final_accuracy_mean', 
                      hue='pruning_applied_epoch', size='mag_prune_fraction', 
                      ax=axes[1], legend='auto', s=150, alpha=0.8, palette='viridis')
        axes[1].errorbar(mag_data['sparsity_pct_mean'], mag_data['final_accuracy_mean'], 
                         yerr=mag_data['final_accuracy_std'], fmt='none', color='grey', alpha=0.5, zorder=-1)
        axes[1].set_title('Magnitude Pruning')
        axes[1].set_xlabel('Weight Sparsity (% Pruned)')
        axes[1].set_ylabel('') # Shared Y axis
        axes[1].grid(True)
        if not np.isnan(baseline_final_acc):
            axes[1].axhline(baseline_final_acc, color='black', linestyle='--', label='Baseline Final Acc')
        axes[1].legend(title='Mag Params', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        axes[1].set_title('Magnitude Pruning (No Data)')
        axes[1].text(0.5, 0.5, 'No Magnitude Data Found', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    save_path = os.path.join(output_dir, f'{model_name}_FINAL_ACC_vs_SPARSITY.png')
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    plt.close(fig)

def plot_metric_correlations(df_agg, output_dir, model_name, dataset_name):
    """Plots correlation heatmaps between key aggregated metrics for each method."""
    set_plot_style()

    methods = {
        'Baseline': df_agg[(df_agg['model'] == model_name) & (df_agg['pruning_method'] == 'none')],
        'NGF': df_agg[(df_agg['model'] == model_name) & (df_agg['pruning_method'].isin(['ngf', 'ngf_scheduled']))],
        'Magnitude': df_agg[(df_agg['model'] == model_name) & (df_agg['pruning_method'] == 'magnitude')]
    }

    # Define metrics to correlate (using mean aggregated values)
    # Note: final_pruning_rate is NaN for baseline, so correlation involving it won't work there.
    metrics_to_correlate = [
        'best_accuracy_mean', 
        'final_pruning_rate_mean', # Sparsity
        'training_time_mean', 
        'min_test_loss_mean'
        # Could add 'final_accuracy_mean' if aggregated
    ]
    
    # Simplify labels for heatmap axes
    metric_labels = {
        'best_accuracy_mean': 'Best Acc',
        'final_pruning_rate_mean': 'Sparsity',
        'training_time_mean': 'Train Time',
        'min_test_loss_mean': 'Min Loss'
    }

    valid_methods_count = sum(1 for df in methods.values() if not df.empty)
    if valid_methods_count == 0:
        print(f"Skipping Correlation Heatmap for {model_name}: No data found for any method.")
        return
        
    # Determine grid size (e.g., 1 row, N columns)
    n_cols = valid_methods_count
    n_rows = 1 
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), squeeze=False) # Adjust size as needed
    fig.suptitle(f'{model_name.upper()} Metric Correlations ({dataset_name})', fontsize=20, y=1.02)

    plot_idx = 0
    for method_name, method_df in methods.items():
        ax = axes[0, plot_idx] # Assuming 1 row
        
        if method_df.empty or len(method_df) < 2: # Need at least 2 points to correlate
            ax.set_title(f'{method_name} (No/Insufficient Data)')
            ax.text(0.5, 0.5, 'Not enough data\nto calculate correlations.', 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            if plot_idx < valid_methods_count -1 : # Only increment if we plotted something or skipped
                 plot_idx += 1
            continue

        # Select only existing columns and calculate correlation
        valid_metrics = [m for m in metrics_to_correlate if m in method_df.columns]
        if len(valid_metrics) < 2:
             ax.set_title(f'{method_name} (Insufficient Metrics)')
             ax.text(0.5, 0.5, 'Not enough valid metrics\nto calculate correlations.', 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
             ax.set_xticks([])
             ax.set_yticks([])
             if plot_idx < valid_methods_count -1 :
                 plot_idx += 1
             continue
             
        # Handle NaN sparsity for baseline explicitly
        if method_name == 'Baseline':
             current_valid_metrics = [m for m in valid_metrics if m != 'final_pruning_rate_mean']
        else:
             current_valid_metrics = valid_metrics
             
        if len(current_valid_metrics) < 2:
             ax.set_title(f'{method_name} (Insufficient Metrics)')
             ax.text(0.5, 0.5, 'Not enough valid metrics\nto calculate correlations.', 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
             ax.set_xticks([])
             ax.set_yticks([])
             if plot_idx < valid_methods_count -1 :
                 plot_idx += 1
             continue
             
        corr_matrix = method_df[current_valid_metrics].corr()
        
        # Get simplified labels for the current matrix
        current_labels = [metric_labels.get(m, m) for m in current_valid_metrics]

        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                    linewidths=.5, linecolor='black', cbar=True, 
                    xticklabels=current_labels, yticklabels=current_labels, ax=ax)
        ax.set_title(f'{method_name} Correlations')
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
        
        plot_idx += 1
        
    # Hide any unused axes if needed (e.g., if a method had no data)
    while plot_idx < n_cols:
         fig.delaxes(axes[0, plot_idx])
         plot_idx +=1

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout for suptitle
    save_path = os.path.join(output_dir, f'{model_name}_METRIC_CORRELATIONS.png')
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    plt.close()

def plot_convergence_speed(df_raw, output_dir, model_name, dataset_name, baseline_metrics, target_acc_fraction=0.99):
    """Plots the distribution of epochs and time taken to reach a target accuracy."""
    set_plot_style()

    # *** Use 'eval_acc' instead of 'test_accuracy' ***
    required_col = 'eval_acc' 

    if required_col not in df_raw.columns:
        print(f"Skipping Convergence Speed plot for {model_name}: Column '{required_col}' not found in results data.")
        print(f"  (Ensure epoch-wise eval accuracy is saved as '{required_col}' in results.json during training.)")
        return
        
    if baseline_metrics is None or np.isnan(baseline_metrics['accuracy_mean']):
        print(f"Skipping Convergence Speed plot for {model_name}: Baseline accuracy not available.")
        return

    target_accuracy = baseline_metrics['accuracy_mean'] * target_acc_fraction

    # Filter raw data: use required_col
    model_df = df_raw[(
        (df_raw['model'] == model_name) &
        (df_raw['pruning_method'].isin(['ngf', 'ngf_scheduled', 'magnitude'])) &
        # Check if required_col list exists and is not empty in this row
        (df_raw[required_col].apply(lambda x: isinstance(x, list) and len(x) > 0)) &
        (df_raw['training_time'].notna()) &
        (df_raw['training_time'] > 0) &
        (df_raw['epochs_run'].notna()) &
        (df_raw['epochs_run'] > 0)
    )].copy()

    if model_df.empty:
        print(f"Skipping Convergence Speed plot for {model_name}: No valid data with '{required_col}' lists found for NGF/Magnitude rows.")
        return
        
    convergence_data = []
    for _, row in model_df.iterrows():
        method = row['pruning_method']
        method_label = 'NGF' if method in ['ngf', 'ngf_scheduled'] else 'Magnitude'
        run_id = row.get('results_dir', 'unknown')
        epochs_list = row['epoch']
        # *** Use required_col here ***
        accuracies = row[required_col] 
        total_time = row['training_time']
        epochs_completed_count = row['epochs_run']

        if epochs_completed_count <= 0 or total_time <= 0 or not accuracies:
            continue
            
        time_per_epoch = total_time / epochs_completed_count
        
        epoch_reached = -1
        time_reached = -1.0

        min_len = min(len(epochs_list), len(accuracies))
        if min_len != epochs_completed_count:
             print(f"Warning (Convergence): Accuracy ({required_col}) list length ({min_len}) differs from epochs_run ({epochs_completed_count}) for run {run_id}.")

        for i in range(min_len):
            if accuracies[i] >= target_accuracy:
                epoch_reached = epochs_list[i] # Get the actual epoch number
                time_reached = time_per_epoch * (epoch_reached + 1)
                break # Found the first time it reached the target

        if epoch_reached != -1:
            convergence_data.append({
                'Method': method_label,
                'Epochs to Target': epoch_reached + 1, # Report 1-based epoch count
                'Time to Target (s)': time_reached,
                'Run': run_id
            })
        # else: # Run never reached target accuracy
        #     print(f"Run {run_id} ({method_label}) did not reach target accuracy {target_accuracy:.2f}%")

    if not convergence_data:
        print(f"No runs reached the target accuracy ({target_accuracy:.2f}%) for {model_name}.")
        return

    conv_df = pd.DataFrame(convergence_data)

    # --- Plotting logic remains the same --- 
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    # ... (boxplot/stripplot for Epochs) ...
    # ... (boxplot/stripplot for Time) ...
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, f'{model_name}_CONVERGENCE_SPEED.png')
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    plt.close()

def plot_performance_stability(df_agg, output_dir, model_name, dataset_name):
    """Plots the standard deviation of key metrics across methods."""
    set_plot_style()

    metrics_to_compare = {
        'best_accuracy_std': 'Best Accuracy Std Dev',
        'training_time_std': 'Training Time Std Dev',
        'final_accuracy_std': 'Final Accuracy Std Dev' # Added this one
    }

    plot_data = []
    methods_df = df_agg[df_agg['model'] == model_name].copy()
    if methods_df.empty:
        print(f"Skipping Stability plot for {model_name}: No aggregated data found.")
        return
        
    # Simplify method names
    def simplify_method_stability(method):
         if method in ['ngf', 'ngf_scheduled']: return 'NGF'
         elif method == 'magnitude': return 'Magnitude'
         else: return 'Baseline' # Assumes 'none' is Baseline
    methods_df['Method'] = methods_df['pruning_method'].apply(simplify_method_stability)

    # Consolidate data for each method (take the first row if multiple configs exist per method)
    # Ideally, aggregation should produce one row per (model, method) if not grouping by other hypers.
    # If grouping by hypers, this plot might be less meaningful unless averaged further.
    # Let's average the std dev across configurations for each method.
    agg_stability_data = methods_df.groupby('Method').agg(
         {k: 'mean' for k in metrics_to_compare.keys()} # Average the std devs across configs
    ).reset_index()


    for idx, row in agg_stability_data.iterrows():
        method = row['Method']
        for metric_col, metric_label in metrics_to_compare.items():
            if metric_col in row and not pd.isna(row[metric_col]):
                plot_data.append({
                    'Method': method,
                    'Metric': metric_label,
                    'Standard Deviation': row[metric_col]
                })

    if not plot_data:
        print(f"No standard deviation data found to plot for {model_name}.")
        return

    stability_df = pd.DataFrame(plot_data)

    # --- Plot --- 
    plt.figure(figsize=(12, 7))
    sns.barplot(data=stability_df, x='Metric', y='Standard Deviation', hue='Method',
                palette="colorblind")

    plt.title(f'{model_name.upper()} Performance Stability Across Runs ({dataset_name})')
    plt.xlabel('Performance Metric')
    plt.ylabel('Average Standard Deviation (across seeds/configs)')
    plt.xticks(rotation=15, ha='right')
    plt.grid(True, axis='y')
    plt.legend(title='Pruning Method')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{model_name}_PERFORMANCE_STABILITY.png')
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    plt.close()

def plot_metric_regression(df_agg, output_dir, model_name, dataset_name):
    """Plots scatter plots with regression lines for key metric pairs,
       separating NGF (Neuron/Channel Sparsity) and Magnitude (Weight Sparsity)."""
    set_plot_style()

    # Prepare NGF data
    ngf_data = df_agg[(
        (df_agg['model'] == model_name) &
        (df_agg['pruning_method'].isin(['ngf', 'ngf_scheduled'])) &
        (df_agg['best_accuracy_mean'].notna()) &
        (df_agg['training_time_mean'].notna()) &
        (df_agg['final_pruning_rate_mean'].notna())
    )].copy()
    if not ngf_data.empty:
        ngf_data['sparsity_pct_mean'] = ngf_data['final_pruning_rate_mean'] * 100

    # Prepare Magnitude data
    mag_data = df_agg[(
        (df_agg['model'] == model_name) &
        (df_agg['pruning_method'] == 'magnitude') &
        (df_agg['best_accuracy_mean'].notna()) &
        (df_agg['training_time_mean'].notna()) &
        (df_agg['final_pruning_rate_mean'].notna())
    )].copy()
    if not mag_data.empty:
        mag_data['sparsity_pct_mean'] = mag_data['final_pruning_rate_mean'] * 100

    if ngf_data.empty and mag_data.empty:
        print(f"Skipping Metric Regression plot for {model_name}: No valid NGF or Magnitude data found.")
        return

    # --- Plot (2x2 Grid) --- 
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), sharex='col') # Share X within columns somewhat meaningless here due to diff sparsity types
    fig.suptitle(f'{model_name.upper()} Metric Relationships with Regression ({dataset_name})', fontsize=20, y=1.03)

    # --- Row 1: Best Accuracy vs. Sparsity --- 
    # NGF Panel (Top Left)
    ax1 = axes[0, 0]
    if not ngf_data.empty:
        sns.regplot(data=ngf_data, x='sparsity_pct_mean', y='best_accuracy_mean', ax=ax1,
                    scatter_kws={'s': 80, 'alpha': 0.6}, color=sns.color_palette("colorblind")[0])
        ax1.set_title('NGF: Accuracy vs. Sparsity')
        ax1.set_xlabel('Neuron/Channel Sparsity (%)')
        ax1.set_ylabel('Avg. Best Accuracy (%)')
        ax1.grid(True)
    else:
        ax1.set_title('NGF: Accuracy vs. Sparsity (No Data)')
        ax1.text(0.5, 0.5, 'No NGF Data', ha='center', va='center', transform=ax1.transAxes)

    # Magnitude Panel (Top Right)
    ax2 = axes[0, 1]
    if not mag_data.empty:
        sns.regplot(data=mag_data, x='sparsity_pct_mean', y='best_accuracy_mean', ax=ax2,
                    scatter_kws={'s': 80, 'alpha': 0.6}, color=sns.color_palette("colorblind")[1])
        ax2.set_title('Magnitude: Accuracy vs. Sparsity')
        ax2.set_xlabel('Weight Sparsity (%)')
        ax2.set_ylabel('') # Share Y axis label maybe?
        ax2.grid(True)
    else:
        ax2.set_title('Magnitude: Accuracy vs. Sparsity (No Data)')
        ax2.text(0.5, 0.5, 'No Magnitude Data', ha='center', va='center', transform=ax2.transAxes)
        
    # --- Row 2: Training Time vs. Sparsity --- 
    # NGF Panel (Bottom Left)
    ax3 = axes[1, 0]
    if not ngf_data.empty:
        sns.regplot(data=ngf_data, x='sparsity_pct_mean', y='training_time_mean', ax=ax3,
                    scatter_kws={'s': 80, 'alpha': 0.6}, color=sns.color_palette("colorblind")[0])
        ax3.set_title('NGF: Training Time vs. Sparsity')
        ax3.set_xlabel('Neuron/Channel Sparsity (%)')
        ax3.set_ylabel('Avg. Training Time (s)')
        ax3.grid(True)
    else:
        ax3.set_title('NGF: Training Time vs. Sparsity (No Data)')
        ax3.text(0.5, 0.5, 'No NGF Data', ha='center', va='center', transform=ax3.transAxes)

    # Magnitude Panel (Bottom Right)
    ax4 = axes[1, 1]
    if not mag_data.empty:
        sns.regplot(data=mag_data, x='sparsity_pct_mean', y='training_time_mean', ax=ax4,
                    scatter_kws={'s': 80, 'alpha': 0.6}, color=sns.color_palette("colorblind")[1])
        ax4.set_title('Magnitude: Training Time vs. Sparsity')
        ax4.set_xlabel('Weight Sparsity (%)')
        ax4.set_ylabel('') # Share Y axis label?
        ax4.grid(True)
    else:
        ax4.set_title('Magnitude: Training Time vs. Sparsity (No Data)')
        ax4.text(0.5, 0.5, 'No Magnitude Data', ha='center', va='center', transform=ax4.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout for suptitle
    save_path = os.path.join(output_dir, f'{model_name}_METRIC_REGRESSION.png')
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    plt.close()

def plot_ngf_loss_curves(df_raw, output_dir, model_name, dataset_name):
    """Plots training and testing loss curves specifically for NGF runs,
       differentiating runs by hyperparameters."""
    set_plot_style()

    # Filter raw data for the model and NGF methods, requiring epoch data
    ngf_df = df_raw[
        (df_raw['model'] == model_name) &
        (df_raw['pruning_method'].isin(['ngf', 'ngf_scheduled'])) &
        (df_raw['train_loss'].apply(lambda x: isinstance(x, list) and len(x) > 0)) &
        (df_raw['test_loss'].apply(lambda x: isinstance(x, list) and len(x) > 0)) &
        (df_raw['epoch'].apply(lambda x: isinstance(x, list) and len(x) > 0))
    ].copy()

    if ngf_df.empty:
        print(f"Skipping NGF Loss Curve plot for {model_name}: No valid NGF epoch data found.")
        return

    # Unpack epoch data and create unique run identifiers
    loss_data = []
    max_epochs = 0
    for _, row in ngf_df.iterrows():
        run_id = row.get('results_dir', 'unknown')
        epochs = row['epoch']
        train_losses = row['train_loss']
        test_losses = row['test_loss']

        # Create a label based on NGF hyperparams
        start = row.get('ngf_start_epoch', 'N/A')
        duration = row.get('ngf_duration', 'N/A')
        keep_frac = row.get('ngf_keep_fraction', 'N/A')
        try:
            keep_frac_str = f"{keep_frac:.2f}" if isinstance(keep_frac, (int, float)) else str(keep_frac)
        except TypeError:
            keep_frac_str = str(keep_frac) # Fallback if formatting fails
         
        # Store numeric keep_fraction and duration for plotting aesthetics
        try:
            numeric_keep_frac = float(keep_frac) if keep_frac != 'N/A' else np.nan
        except (ValueError, TypeError):
            numeric_keep_frac = np.nan
        try:
            numeric_duration = int(duration) if duration != 'N/A' else np.nan
        except (ValueError, TypeError):
            numeric_duration = np.nan

        # Simpler label: Focus on key hyperparameters
        run_label = f"Start:{start}_Dur:{duration}_Keep:{keep_frac_str}"

        min_len = min(len(epochs), len(train_losses), len(test_losses))
        if min_len != len(epochs):
             print(f"Warning (NGF Loss): Loss list length mismatch for run {run_id}. Truncating to {min_len} epochs.")

        for i in range(min_len):
            loss_data.append({
                'Keep Fraction': numeric_keep_frac,
                'Duration': numeric_duration,
                'Start Epoch': start, # Keep for potential filtering/tooltips later
                'Epoch': epochs[i],
                'Loss Type': 'Train',
                'Loss': train_losses[i],
                'Run ID': run_id # Keep for grouping if needed
            })
            loss_data.append({
                'Keep Fraction': numeric_keep_frac,
                'Duration': numeric_duration,
                'Start Epoch': start,
                'Epoch': epochs[i],
                'Loss Type': 'Test',
                'Loss': test_losses[i],
                'Run ID': run_id
            })
        if min_len > max_epochs:
            max_epochs = min_len

    if not loss_data:
        print(f"No processable NGF loss data found for {model_name} after unpacking.")
        return

    loss_df = pd.DataFrame(loss_data)

    # --- Plot ---
    # Use relplot for separate Train/Test facets, map run label to hue
    g = sns.relplot(data=loss_df, x='Epoch', y='Loss', 
                    hue='Keep Fraction', # Color gradient for keep fraction
                    style='Duration',    # Line style for duration
                    col='Loss Type',
                    kind='line', errorbar=None, # Show individual lines
                    facet_kws={'sharey': False},
                    palette='viridis', # Use a sequential palette for hue
                    height=6, aspect=1.2,
                    legend='full') # Use 'full' legend for more detail

    g.fig.suptitle(f'{model_name.upper()} NGF Loss Curves ({dataset_name})', y=1.03)
    g.set_axis_labels('Epoch', 'Loss')
    g.set_titles("{col_name} Loss")
    
    # Adjust legend location (optional, try defaults first)
    # sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))


    for ax in g.axes.flat:
        ax.grid(True, linestyle='--', alpha=0.6)
        # Optional: Set xlim based on max_epochs across runs
        # ax.set_xlim(0, max_epochs) 

    plt.tight_layout() # Use default tight_layout first
    save_path = os.path.join(output_dir, f'{model_name}_NGF_LOSS_CURVES.png')
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    plt.close()

def create_comparison_plots(df_agg, df_raw, output_dir, model_name, dataset_name):
    """Generates a set of comparison plots for a specific model."""
    set_plot_style()
    baseline_metrics = get_baseline_metrics(df_agg, model_name)

    # Prepare dataframes, calculate sparsity percentages first
    ngf_data = df_agg[(df_agg['model'] == model_name) & (df_agg['pruning_method'].isin(['ngf', 'ngf_scheduled']))].copy()
    if not ngf_data.empty:
        ngf_data['sparsity_pct_mean'] = ngf_data['final_pruning_rate_mean'] * 100
        ngf_data['sparsity_pct_std'] = ngf_data['final_pruning_rate_std'] * 100 # For potential x-error bars

    mag_data = df_agg[(df_agg['model'] == model_name) & (df_agg['pruning_method'] == 'magnitude')].copy()
    if not mag_data.empty:
         mag_data['sparsity_pct_mean'] = mag_data['final_pruning_rate_mean'] * 100
         mag_data['sparsity_pct_std'] = mag_data['final_pruning_rate_std'] * 100 # For potential x-error bars

    # --- Plot 1 & 2: Accuracy vs. Sparsity (NGF vs Magnitude) --- 
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    fig.suptitle(f'{model_name.upper()} Accuracy vs. Sparsity ({dataset_name}) - Note Sparsity Type Difference', fontsize=20)
    
    # NGF Panel
    if not ngf_data.empty:
        # Use the calculated percentage column for x
        sns.scatterplot(data=ngf_data, x='sparsity_pct_mean', y='best_accuracy_mean', 
                      size='ngf_keep_fraction', hue='ngf_duration', style='ngf_start_epoch',
                      ax=axes[0], legend='auto', s=150, alpha=0.8)
        # Use the calculated percentage column for errorbar x position
        axes[0].errorbar(ngf_data['sparsity_pct_mean'], ngf_data['best_accuracy_mean'], 
                         yerr=ngf_data['best_accuracy_std'], fmt='none', color='grey', alpha=0.5, zorder=-1)
        axes[0].set_title('NGF Pruning')
        axes[0].set_xlabel('Neuron/Channel Sparsity (% Pruned)')
        axes[0].set_ylabel('Best Test Accuracy (%)')
        axes[0].grid(True)
        if baseline_metrics and not np.isnan(baseline_metrics['accuracy_mean']):
            axes[0].axhline(baseline_metrics['accuracy_mean'], color='black', linestyle='--', label='Baseline Acc')
        axes[0].legend(title='NGF Params', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        axes[0].set_title('NGF Pruning (No Data)')
        axes[0].text(0.5, 0.5, 'No NGF Data Found', horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)

    # Magnitude Panel
    if not mag_data.empty:
        # Use the calculated percentage column for x
        sns.scatterplot(data=mag_data, x='sparsity_pct_mean', y='best_accuracy_mean', 
                      hue='pruning_applied_epoch', size='mag_prune_fraction', 
                      ax=axes[1], legend='auto', s=150, alpha=0.8, palette='viridis')
        # Use the calculated percentage column for errorbar x position
        axes[1].errorbar(mag_data['sparsity_pct_mean'], mag_data['best_accuracy_mean'], 
                         yerr=mag_data['best_accuracy_std'], fmt='none', color='grey', alpha=0.5, zorder=-1)
        axes[1].set_title('Magnitude Pruning')
        axes[1].set_xlabel('Weight Sparsity (% Pruned)')
        axes[1].set_ylabel('') # Shared Y axis
        axes[1].grid(True)
        if baseline_metrics and not np.isnan(baseline_metrics['accuracy_mean']):
            axes[1].axhline(baseline_metrics['accuracy_mean'], color='black', linestyle='--', label='Baseline Acc')
        axes[1].legend(title='Mag Params', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        axes[1].set_title('Magnitude Pruning (No Data)')
        axes[1].text(0.5, 0.5, 'No Magnitude Data Found', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    save_path = os.path.join(output_dir, f'{model_name}_ACC_vs_SPARSITY.png')
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    plt.close(fig)

    # --- Plot 3: Training Time & Min Loss Distribution (using df_raw) --- 
    df_plot = df_raw[df_raw['model'] == model_name].copy()
    if df_plot.empty:
        print(f"No raw data found for model {model_name} for distribution plots.")
        return # Skip remaining plots if no raw data
        
    # Create simplified method labels for plotting
    def simplify_method(row):
        if row['pruning_method'] == 'ngf' or row['pruning_method'] == 'ngf_scheduled':
            return 'NGF'
        elif row['pruning_method'] == 'magnitude':
            return 'Magnitude'
        else:
            return 'Baseline'
    df_plot['Method'] = df_plot.apply(simplify_method, axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f'{model_name.upper()} Performance Distribution ({dataset_name})', fontsize=20)
    
    # Training Time Box Plot
    sns.boxplot(data=df_plot, x='Method', y='training_time', ax=axes[0])
    sns.stripplot(data=df_plot, x='Method', y='training_time', ax=axes[0], color='.3', alpha=0.5)
    axes[0].set_title('Training Time Distribution')
    axes[0].set_xlabel('Pruning Method')
    axes[0].set_ylabel('Total Training Time (seconds)')
    axes[0].grid(True)

    # Min Test Loss Box Plot
    sns.boxplot(data=df_plot, x='Method', y='min_test_loss', ax=axes[1])
    sns.stripplot(data=df_plot, x='Method', y='min_test_loss', ax=axes[1], color='.3', alpha=0.5)
    axes[1].set_title('Minimum Test Loss Distribution')
    axes[1].set_xlabel('Pruning Method')
    axes[1].set_ylabel('Min Test Loss (Avg/Batch)')
    axes[1].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, f'{model_name}_DISTRIBUTIONS_Time_Loss.png')
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    plt.close(fig)

def main(args):
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory '{args.results_dir}' not found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and process data
    df_raw = load_results(args.results_dir, args.dataset)
    if df_raw.empty:
        return

    # Aggregate summary stats
    df_agg = aggregate_results(df_raw)
    
    # --- START RIGOROUS DIAGNOSTIC CHECK --- 
    print("\n--- Running Diagnostic Checks ---")
    
    # Check 1: Raw NGF data with zero sparsity
    ngf_raw_df = df_raw[df_raw['pruning_method'].isin(['ngf', 'ngf_scheduled'])].copy()
    suspicious_ngf_raw = ngf_raw_df[np.isclose(ngf_raw_df['final_pruning_rate'].fillna(-1), 0)]
    
    if not suspicious_ngf_raw.empty:
        print(f"\n[WARNING] Found {len(suspicious_ngf_raw)} RAW NGF run(s) with final_pruning_rate near 0:")
        cols_to_show = ['model', 'seed', 'pruning_method', 'ngf_keep_fraction', 'final_pruning_rate', 'results_dir']
        # Ensure columns exist before trying to print
        cols_to_show = [col for col in cols_to_show if col in suspicious_ngf_raw.columns]
        print(suspicious_ngf_raw[cols_to_show].to_string())
        print("  -> Check the results.json in the listed 'results_dir' to verify saved 'final_pruning_rate'.")
    else:
        print("\n[OK] No raw NGF runs found with final_pruning_rate near 0.")
        
    # Check 2: Aggregated NGF data with zero mean sparsity
    if not df_agg.empty:
        ngf_agg_df = df_agg[df_agg['pruning_method'].isin(['ngf', 'ngf_scheduled'])].copy()
        # Ensure the mean column exists before checking
        if 'final_pruning_rate_mean' in ngf_agg_df.columns:
             suspicious_ngf_agg = ngf_agg_df[np.isclose(ngf_agg_df['final_pruning_rate_mean'].fillna(-1), 0)]
             if not suspicious_ngf_agg.empty:
                 print(f"\n[WARNING] Found {len(suspicious_ngf_agg)} AGGREGATED NGF configuration(s) with final_pruning_rate_mean near 0:")
                 cols_to_show_agg = ['model', 'pruning_method', 'ngf_keep_fraction', 'final_pruning_rate_mean', 'run_count']
                 cols_to_show_agg = [col for col in cols_to_show_agg if col in suspicious_ngf_agg.columns]
                 print(suspicious_ngf_agg[cols_to_show_agg].to_string())
                 print("  -> This might indicate an issue with saved data in underlying runs or the aggregation logic.")
             else:
                 print("\n[OK] No aggregated NGF configurations found with final_pruning_rate_mean near 0.")
        else:
            print("\n[INFO] Column 'final_pruning_rate_mean' not found in aggregated data for NGF checks.")
    else:
        print("\n[INFO] Aggregated dataframe is empty, skipping aggregated checks.")
        
    print("--- End Diagnostic Checks ---\n")
    # --- END RIGOROUS DIAGNOSTIC CHECK ---

    # Original plotting logic continues below...
    print("Aggregated Data Sample (Top 5 rows):")
    print(df_agg.head().to_string())
    if df_agg.empty:
        print("Aggregated data is empty, cannot proceed with plotting.")
        return

    # Get unique models
    models = df_raw['model'].unique()
    for model in models:
        if pd.isna(model): continue
        print(f"\n--- Generating Plots for Model: {model} ---")
        
        baseline_metrics = get_baseline_metrics(df_agg, model)

        # Generate the original comparison plots
        create_comparison_plots(df_agg, df_raw, args.output_dir, model, args.dataset)
        
        # Generate the Accuracy Drop plot
        plot_accuracy_drop_vs_sparsity(df_agg, args.output_dir, model, args.dataset, baseline_metrics)
        
        # Generate the Layer-wise Sparsity plot
        plot_layerwise_sparsity(df_raw, args.output_dir, model, args.dataset)
        
        # Generate the Accuracy vs Time plot
        plot_accuracy_time_frontier(df_agg, args.output_dir, model, args.dataset, baseline_metrics)
        
        # Generate the Loss Curves plot
        plot_loss_curves(df_raw, args.output_dir, model, args.dataset)
        
        # Generate Accuracy Distribution plot
        plot_accuracy_distribution(df_raw, args.output_dir, model, args.dataset)
        
        # Generate NGF Sensitivity plot
        plot_ngf_sensitivity(df_agg, args.output_dir, model, args.dataset)
        
        # Generate Sparsity Comparison plot
        plot_sparsity_for_target_accuracy(df_raw, df_agg, args.output_dir, model, args.dataset, baseline_metrics)
        
        # Generate Loss vs Time plot
        plot_loss_vs_time(df_raw, args.output_dir, model, args.dataset)
        
        # Generate Final Acc vs Sparsity plot
        plot_final_accuracy_vs_sparsity(df_agg, args.output_dir, model, args.dataset)
        
        # Generate Metric Correlations plot
        plot_metric_correlations(df_agg, args.output_dir, model, args.dataset)
        
        # Generate Convergence Speed plot
        plot_convergence_speed(df_raw, args.output_dir, model, args.dataset, baseline_metrics)
        
        # Generate Performance Stability plot
        plot_performance_stability(df_agg, args.output_dir, model, args.dataset)
        
        # Generate Metric Regression plot
        plot_metric_regression(df_agg, args.output_dir, model, args.dataset)

        # Add call to the new NGF loss plot function
        plot_ngf_loss_curves(df_raw, args.output_dir, model, args.dataset)

    print(f"\nAll comparison plots saved to '{args.output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot comparison results from pruning experiments.')
    parser.add_argument('--results-dir', type=str, default='results', help='Base directory containing dataset/model subdirectories.')
    parser.add_argument('--output-dir', type=str, default='results/comparison_plots', help='Directory to save comparison plots.')
    parser.add_argument('--dataset', type=str, default='fashion_mnist', help='Dataset results to plot (e.g., fashion_mnist)')
    args = parser.parse_args()
    main(args) 