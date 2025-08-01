import gymnasium as gym

def get_cartpole_env(render_mode="rgb_array"):
    """
    Initializes CartPole-v1 environment.

    Args:
        render_mode (str): Rendering mode for the environment.
                           "rgb_array" is needed for video saving.
                           "human" for displaying to screen.

    Returns:
        gym.Env: The initialized CartPole environment.
    """
    env = gym.make("CartPole-v1", render_mode=render_mode)
    print(f"Initialized CartPole-v1 environment with render_mode='{render_mode}'")
    return env 