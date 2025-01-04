import gym
import numpy as np
from ppo import PPO  # Make sure 'ppo.py' is in the same folder or in your Python path

def train_ppo():
    # 1. Create the environment
    env = gym.make("Pendulum-v1")
    
    # 2. Instantiate your PPO agent
    #    Pass in any hyperparameters you need. Some are shown here as an example.
    agent = PPO(
        ob_space=3,
        actions=1,
        n_batches=10,
        gamma=0.99,
        lam=0.95,
        kl_coeff=0.2,
        clip_rewards=False,
        clip_param=0.2,
        vf_clip_param=10.0,
        entropy_coeff=0.01,
        a_lr=3e-4,
        c_lr=3e-4,
        device="cpu",
        max_ts=1_000_000,

        # Any custom kwargs can also be passed in here. For example:
        timesteps_per_batch=3,
        max_timesteps_per_episode=200,
        n_updates_per_iteration=3,
    )
    
    # 3. Train the agent
    total_timesteps = 50_000  # Decide how long you want to train
    agent.learn(total_timesteps=total_timesteps, env=env)
    
    # 4. (Optional) Close the environment
    env.close()

if __name__ == "__main__":
    train_ppo()
