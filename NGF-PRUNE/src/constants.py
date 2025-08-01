"""
Constants for NGF-PRUNE project.
"""

# Data constants
DATA_DIR = './data'
MNIST_NORMALIZE_MEAN = (0.1307,)
MNIST_NORMALIZE_STD = (0.3081,)
FASHION_MNIST_NORMALIZE_MEAN = (0.2860,)
FASHION_MNIST_NORMALIZE_STD = (0.3530,)

# Model architecture constants
SIMPLE_MLP_DEFAULTS = {
    'input_dim': 784,
    'hidden_dim1': 512,
    'hidden_dim2': 256,
    'output_dim': 10
}

LENET5_DEFAULTS = {
    'num_classes': 10,
    'conv1_out_channels': 6,
    'conv2_out_channels': 16,
    'conv_kernel_size': 5,
    'fc1_input_size': 16 * 5 * 5,  # 400
    'fc1_output_size': 120,
    'fc2_output_size': 84
}

POLICY_MLP_DEFAULTS = {
    'input_dim': 4,  # CartPole state dimension
    'hidden_dim': 128,
    'output_dim': 2  # CartPole action dimension
}

# Training constants
DEFAULT_BATCH_SIZE = 128
DEFAULT_TEST_BATCH_SIZE = 1000
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_GAMMA = 0.99  # RL discount factor
DEFAULT_REWARD_WINDOW_SIZE = 100  # Episodes to average for RL reward window

# Pruning constants
DEFAULT_PRUNING_THRESHOLD = 0.7
MIN_KEEP_FRACTION = 0.0
MAX_KEEP_FRACTION = 1.0
MIN_PRUNE_FRACTION = 0.0
MAX_PRUNE_FRACTION = 1.0

# File extensions and directories
RESULTS_DIR = 'results'
VIDEO_DIR = 'videos'
CONFIG_FILENAME = 'config.json'
RESULTS_FILENAME = 'results.json'
MODEL_FILENAME = 'final_model.pth'
PLOT_FILENAME_SUPERVISED = 'training_plots.png'
PLOT_FILENAME_RL = 'rl_training_plots.png'
PLOT_FILENAME_SCHEDULED = 'scheduled_run_summary.png'

# Environment constants
CARTPOLE_ENV_NAME = 'CartPole-v1'
RENDER_MODE_RGB = 'rgb_array'
RENDER_MODE_HUMAN = 'human'

# Supported choices for arguments
SUPPORTED_TASKS = ['supervised', 'rl']
SUPPORTED_DATASETS = ['mnist', 'fashion_mnist']
SUPPORTED_MODELS = ['mlp', 'lenet']
SUPPORTED_PRUNING_METHODS = ['none', 'ngf', 'magnitude', 'ngf_scheduled']
SUPPORTED_RL_ENVS = ['CartPole-v1']

# Error messages
ERROR_INVALID_DATASET = "Unknown dataset: {}"
ERROR_INVALID_MODEL = "Unknown model type: {}"
ERROR_INVALID_TASK = "Unknown task type: {}"
ERROR_INVALID_KEEP_FRACTION = "Pruning threshold (keep_fraction) must be (0.0, 1.0] for NGF."
ERROR_INVALID_PRUNE_FRACTION = "Pruning threshold (prune_fraction) must be [0.0, 1.0) for magnitude."
ERROR_INVALID_CRITICAL_PERIOD = "Critical duration must be positive for NGF pruning."
ERROR_INVALID_APPLY_EPOCH = "Pruning apply epoch must be positive for magnitude pruning."
ERROR_INVALID_SCHEDULE_FRACTIONS = "Keep fractions invalid for schedule."
ERROR_INVALID_SCHEDULE_EPOCHS = "Invalid start/end epochs for schedule."
ERROR_INVALID_PRUNING_FREQUENCY = "Pruning frequency must be positive."