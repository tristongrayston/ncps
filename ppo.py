import torch as t
import torch.nn as nn 
from torch import optim 
import numpy as np
import time
import os.path as osp

'''
Todo: Add Entropy
'''


class PPO:
    def __init__(self, 
                 ob_space, 
                 actions, 
                 n_batches, 
                 gamma, 
                 lam, 
                 kl_coeff, 
                 clip_rewards, 
                 clip_param, 
                 vf_clip_param, 
                 entropy_coeff,
                 a_lr,
                 c_lr,
                 device,
                 **kwargs):
        """
        Proximal Policy Optimization (PPO) initialization.
        
        Parameters
        ----------
        ob_space : int or tuple
            The dimensionality (or shape) of the observation space.
        actions : int or tuple
            The dimensionality (or shape) of the action space.
        n_batches : int
            The number of batches for training updates.
        gamma : float
            Discount factor for rewards.
        lam : float
            Lambda for GAE (Generalized Advantage Estimation).
        kl_coeff : float
            Coefficient for KL divergence (used in early stopping).
        clip_rewards : bool
            Whether to clip rewards.
        clip_param : float
            Clipping parameter for PPO (policy clipping).
        vf_clip_param : float
            Clipping parameter for value function updates.
        entropy_coeff : float
            Coefficient for entropy regularization.
        device : obj
            Device which holds data.
        (the two LR's): float
            a_lr = actor lr
            c_lr = critic lr
        **kwargs : dict
            Additional keyword arguments.
        """
        self.ob_space = ob_space
        self.actions = actions
        self.n_batches = n_batches
        self.gamma = gamma
        self.lam = lam
        self.kl_coeff = kl_coeff
        self.clip_rewards = clip_rewards
        self.clip_param = clip_param
        self.vf_clip_param = vf_clip_param
        self.entropy_coeff = entropy_coeff
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.device = device

        # Optionally store any extra keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Same backbone for shared feature identification 
        self.backbone = t.Sequential([
            nn.Linear(ob_space, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        ]).to(self.device)

        # Actor
        self.actor = t.Sequential([
            nn.Linear(64, actions),
            nn.Softmax()
        ]).to(self.device)

        # Critic
        self.critic = t.Sequential([
            nn.Linear(64, 1)
        ]).to(self.device)

        # Get Parameters 
        actor_params = list(self.backbone.parameters()) + list(self.actor.parameters())
        critic_params = list(self.backbone.parameters()) + list(self.critic.parameters())

        # Optimizers
        self.actor_optim = optim.Adam(actor_params, lr=a_lr)
        self.critic_optim = optim.Adam(critic_params, lr=c_lr)

        self.cov_var = t.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = t.diag(self.cov_var)

    def get_action(self, obs):
        """
            Queries an action from the actor network, should be called from rollout.

            Parameters:
                obs - the observation at the current timestep

            Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """
        # This might be wrong, check on this later
        with t.no_grad():
            feats = self.backbone(obs)
            mean = self.actor(feats)

        dist = t.distributions.MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()

        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()
    
    def get_vf(self, obs):
        with t.no_grad():
            feats = self.backbone(obs)
            vf = self.critic(feats)

        return vf.detach().numpy()
    
    def calculate_gaes(self, rewards, values, gamma=0.99, decay=0.97):
        """
        Return the General Advantage Estimates from the given rewards and values.
        Paper: https://arxiv.org/pdf/1506.02438.pdf
        Credit: Eden Meyer
        """
        # This is only important if you're running on GPU
        values = t.stack(values).detach().cpu().numpy()

        next_values = np.concatenate([values[1:], [[0]]])
        deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

        gaes = [deltas[-1]]
        for i in reversed(range(len(deltas)-1)):
            gaes.append(deltas[i] + decay * gamma * gaes[-1])

        return np.array(gaes[::-1])


    def rollout(self, env, max_steps):
        """
        Takes the environment and performs one episode of the environment. 
        """
        train_data = [[], [], [], [], [], []] # obs, action, rewards, values, act_log_probs, dones
        obs, _ = env.reset()

        ep_reward = 0.0

        # Perform Rollout 
        for _ in range(max_steps):
            # Action
            vect_obs = t.tensor(obs, dtype=t.float32, device=self.device)

            action, log_prob = self.get_action(vect_obs)
            vals = self.get_vf(vect_obs)

            next_obs, reward, done, trun, _ = env.step(action.item())
            for i, item in enumerate((obs, action, reward, vals, log_prob, done)):
                train_data[i].append(item)

            obs = next_obs
            ep_reward += reward 
            if done:
                break
        
        # --- Get GAE, replacing values with advantages. --- 
        
        train_data[3] = self.calculate_gaes(train_data[2], train_data[3])
        return train_data, ep_reward


