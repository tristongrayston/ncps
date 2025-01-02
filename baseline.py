import torch
import matplotlib.pyplot as plt
import gym
from torch.utils.data import Dataset
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPO

# Environment
env = gym.make("HalfCheetah-v5")

# Agent
agent = PPO()