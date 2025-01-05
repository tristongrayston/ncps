import torch as t
import torch.nn as nn 
from torch import optim 
import numpy as np
import time
import os.path as osp

'''
Todo: Add Entropy
TODO: Note, for pendulum, the episodes trunicate at 200 time steps and has no failure condition. 
TODO: Turns out, moving things to gpu doesn't make things quicker if you have a small enough model such that
the transfer from gpu -> cpu is outpaced by the time it takes for your data to go through your model. 
Making this quicker, then, is for a time when we can run multiple envs at the same time. until then, cpu it is! 
'''
t.autograd.set_detect_anomaly(True)

class PPO:
    def __init__(
        self,
        ob_space,
        actions,
        n_batches=10,            # default number of batches
        gamma=0.99,              # discount factor
        lam=0.95,                # GAE (Generalized Advantage Estimation) lambda
        kl_coeff=0.2,            # coefficient for KL penalty (if used)
        clip_rewards=False,      # whether to clip rewards
        clip_param=0.2,          # PPO clipping parameter
        vf_clip_param=10.0,      # value function clipping parameter
        entropy_coeff=0.01,      # entropy bonus coefficient
        a_lr=1e-5,               # actor learning rate
        c_lr=1e-5,               # critic learning rate
        device='cpu',            # device, e.g. 'cpu' or 'cuda'
        max_ts=1_000_000,        # max timesteps
        **kwargs
        ):
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
        self.clip = clip_param
        self.vf_clip_param = vf_clip_param
        self.entropy_coeff = entropy_coeff
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.max_ts = max_ts
        self.device = device

        # Optionally store any extra keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Same backbone for shared feature identification 
        self.backbone = nn.Sequential(
            nn.Linear(ob_space, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        ).to(self.device)
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(64, actions),
            nn.Tanh()
        ).to(self.device)

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(64, 1)
        ).to(self.device)

        # init orth. weights
        self.backbone.apply(self.init_weights)
        self.actor.apply(self.init_weights)
        self.critic.apply(self.init_weights)

        # Get Parameters 
        actor_params = list(self.backbone.parameters()) + list(self.actor.parameters())
        critic_params = list(self.backbone.parameters()) + list(self.critic.parameters())

        # Optimizers
        self.actor_optim = optim.Adam(actor_params, lr=a_lr, eps=1e-5)
        self.critic_optim = optim.Adam(critic_params, lr=c_lr, eps=1e-5)

        self.cov_var = t.full(size=(self.actions,), fill_value=0.5).to(self.device)
        self.cov_mat = t.diag(self.cov_var).to(self.device)

        # Logger, credit: Eric Yang Yu
        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,          # timesteps so far
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
            'actor_losses': [],     # losses of actor network in current iteration
        }

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            t.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def get_action(self, obs, rollout=True):
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
            mean = self.actor(feats)*2

        dist = t.distributions.MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()

        log_prob = dist.log_prob(action)

        return action.cpu().numpy(), log_prob
    
    def get_vf(self, obs, rollout=True):
        with t.no_grad():
            feats = self.backbone(obs)
            vf = self.critic(feats)

        return vf.cpu().numpy()
    
    def evaluate(self, batch_obs, batch_acts):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.

            Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)

            Return:
                V - the predicted values of batch_obs
                log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
        """
        feats = self.backbone(batch_obs)

        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = self.critic(feats).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(feats)*2
        dist = t.distributions.MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        entropy = dist.entropy().mean()

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs, entropy
    
    def calculate_gaes(self, rewards, values):
        """
        Return the General Advantage Estimates from the given rewards and values.
        Paper: https://arxiv.org/pdf/1506.02438.pdf
        Credit: Eden Meyer
        """
        # This is only important if you're running on GPU
        #values = t.stack(values).detach().cpu().numpy()

        next_values = np.concatenate([values[1:], [[0]]])
        deltas = [rew + self.gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

        gaes = [deltas[-1]]
        for i in reversed(range(len(deltas)-1)):
            gaes.append(deltas[i] + self.lam * self.gamma * gaes[-1])

        return np.array(gaes[::-1])
    
    def discount_rewards(self, rewards):
        """
        Return discounted rewards based on the given rewards and gamma param.

        Credit: Eden Meyer
        """
        new_rewards = [float(rewards[-1])]
        for i in reversed(range(len(rewards)-1)):
            new_rewards.append(float(rewards[i]) + self.gamma * new_rewards[-1])
        return t.tensor(new_rewards[::-1], dtype=t.float32, device=self.device)


    def rollout(self, env):
        """
        Takes the environment and performs one episode of the environment. 
        """
        b_obs = []
        actions = []
        advantages = []
        returns = []
        act_log_probs = []
        dones = []
        ep_rewards = []

        obs, _ = env.reset()

        
        for i in range(self.timesteps_per_batch):
            # We want to make sure we train on atleast this many number of timesteps per episode
            rollout_reward = []
            rollout_values = []
            done = False
            obs, _ = env.reset()
            ep_reward = 0.0
            
            # Perform Rollout 
            for _ in range(self.max_timesteps_per_episode):
                # Action
                vect_obs = t.tensor(obs, dtype=t.float32, device=self.device)

                action, log_prob = self.get_action(vect_obs)
                vals = self.get_vf(vect_obs)

                #print("action", action, " item now ", action.item())

                next_obs, reward, term, trun, _ = env.step(action)

                done = term | trun 

                for dest_list, new_value in zip(
                    (b_obs, actions, rollout_reward, rollout_values, act_log_probs, dones),
                    (vect_obs, action, reward, vals, log_prob, done)):

                    #print("dest list$$$$: ", dest_list, "and now type ", type(dest_list))
                    dest_list.append(new_value)
                    rollout_values.append(vals)
                    dones.append(done)

                obs = next_obs
                ep_reward += reward 
                if done:
                    break
        
            # Get GAE, replacing values with advantages.
        
            rollout_adv = self.calculate_gaes(rollout_reward, rollout_values)
            advantages.append(rollout_adv)
            ep_rewards.append(ep_reward)

            # Get returns

            ep_returns = self.discount_rewards(rollout_reward)
            returns.append(ep_returns)

        # turn things into tensors
        b_obs = t.stack(b_obs, dim=0)
        
        actions = np.stack(actions, axis=0)  
        actions = t.tensor(actions, device=self.device, dtype=t.float32)

        advantages = np.stack(advantages, axis=0)  
        advantages = t.tensor(advantages, device=self.device, dtype=t.float32)
        advantages = advantages.view(-1, 1)

        #act_log_probs = np.stack(act_log_probs, axis=0)  
        act_log_probs = t.stack(act_log_probs, dim=0)

        returns = t.stack(returns, dim=0)
        returns = returns.view(-1, 1)

        act_log_probs = act_log_probs.view(-1, 1)
        
        #print("b_obs shape:", b_obs.shape)
        #print("actions shape:", actions.shape)
        #print("advantages shape:", advantages.shape)
        #print("returns shape:", returns.shape)
        #print("act_log_probs shape:", act_log_probs.shape)
        #print("dones shape : ", sum(dones))


        return b_obs, actions, advantages, returns, act_log_probs, ep_rewards

    def learn(self, total_timesteps, env):
        """
            Train the actor and critic networks. Here is where the main PPO algorithm resides.

            Parameters:
                total_timesteps - the total number of timesteps to train for

            Return:
                None
        """
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0 # Timesteps simulated so far
        i_so_far = 0 # Iterations ran so far
        while t_so_far < total_timesteps:                                                                      
            # Autobots, roll out (just kidding, we're collecting our batch simulations here)
            obs, actions, advantages, returns, act_log_probs, ep_reward = self.rollout(env) 
            
            # Calculate how many timesteps we collected this batch
            t_so_far += obs.shape[0]

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):
                for mb_obs, mb_actions, mb_advantages, mb_returns, mb_act_log_probs in \
                    self.create_minibatches(obs, actions, advantages, returns, act_log_probs, batch_size=500):                                                   
                    # Calculate V_phi and pi_theta(a_t | s_t)
                    V, curr_log_probs, entropy = self.evaluate(mb_obs, mb_actions)

                    V = V.view(-1, 1)
                    curr_log_probs = curr_log_probs.view(-1, 1)

                    ratios = t.exp(curr_log_probs - mb_act_log_probs)

                    # Calculate surrogate losses.
                    surr1 = ratios * mb_advantages
                    surr2 = t.clamp(ratios, 1 - self.clip, 1 + self.clip) * mb_advantages

                    actor_loss = -(t.min(surr1, surr2)).mean() - self.entropy_coeff * entropy
                    critic_loss = nn.MSELoss()(V, mb_returns)

                    # Calculate gradients and perform backward propagation for both network
                    self.actor_optim.zero_grad()
                    self.critic_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    critic_loss.backward()
                    self.actor_optim.step()
                    self.critic_optim.step()

                    # Log actor loss
                    #print("actor loss: ", actor_loss.detach().item())

                self.logger['actor_losses'].append(actor_loss.cpu().detach())

            # Print a summary of our training so far
            self._log_summary(ep_reward, obs.shape[0])

    def create_minibatches(self, obs, actions, advantages, returns, act_log_probs,
                        batch_size=64, shuffle=True):
        """
        Create minibatches from rollout data for PPO or similar RL algorithms.
        
        Parameters
        ----------
        obs : np.ndarray
            Observations from the rollout.
        actions : np.ndarray
            Actions taken during the rollout.
        advantages : np.ndarray
            Computed advantages for each step in the rollout.
        returns : np.ndarray
            Computed returns (discounted sum of rewards) for each step.
        act_log_probs : np.ndarray
            Log probabilities of the actions that were taken.
        ep_rewards : np.ndarray
            Reward values (either per step or per episode, depending on your storage).
        batch_size : int
            Size of each minibatch.
        shuffle : bool
            Whether to shuffle (randomize) the data before splitting into minibatches.
        
        Yields
        ------
        Tuple of np.ndarray
            The next minibatch (obs, actions, advantages, returns, act_log_probs, ep_rewards).
        """
        
        # Number of samples (timesteps) in the rollout
        n_samples = len(obs)
        
        # Optionally shuffle the data (consistent across all arrays)
        if shuffle:
            indices = np.random.permutation(n_samples)
            obs = obs[indices]
            actions = actions[indices]
            advantages = advantages[indices]
            returns = returns[indices]
            act_log_probs = act_log_probs[indices]
        
        # Generate minibatches
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)

            yield (
                obs[start_idx:end_idx],
                actions[start_idx:end_idx],
                advantages[start_idx:end_idx],
                returns[start_idx:end_idx],
                act_log_probs[start_idx:end_idx],
            )


    def _log_summary(self, eps_rewards, no_timesteps):
        """
            Print to stdout what we've logged so far in the most recent batch.

            Parameters:
                None

            Return:
                None
        """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = sum(eps_rewards)/self.timesteps_per_batch
        #avg_actor_loss = no_timesteps/self.timesteps_per_batch

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []



