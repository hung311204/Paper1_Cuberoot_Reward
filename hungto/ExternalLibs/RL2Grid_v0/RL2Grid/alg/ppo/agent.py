from torch.distributions import Categorical, Normal

from hungto.ExternalLibs.RL2Grid_v0.RL2Grid.common.utils import Linear, th_act_fns
from hungto.ExternalLibs.RL2Grid_v0.RL2Grid.common.imports import nn, np, th

class Agent(nn.Module):
    def __init__(self, envs, args, continuous_actions):
        super().__init__()

        critic_layers = args.critic_layers
        act_str, act_fn = args.critic_act_fn, th_act_fns[args.critic_act_fn]
        layers = []
        layers.extend([
            Linear(np.prod(envs.single_observation_space.shape), critic_layers[0], act_str), 
            act_fn
        ])

        for idx, embed_dim in enumerate(critic_layers[1:], start=1): 
            layers.extend([Linear(critic_layers[idx-1], embed_dim, act_str), act_fn])
        layers.append(Linear(critic_layers[-1], 1, 'linear'))
        self.critic = nn.Sequential(*layers)


        actor_layers = args.actor_layers
        act_str, act_fn = args.actor_act_fn, th_act_fns[args.actor_act_fn]
        layers = []
        layers.extend([
            Linear(np.prod(envs.single_observation_space.shape), actor_layers[0], act_str), 
            act_fn
        ])
        
        print("Actor Input shape:", envs.single_observation_space.shape)  # Announce shape of input

        
        for idx, embed_dim in enumerate(actor_layers[1:], start=1): 
            layers.extend([Linear(actor_layers[idx-1], embed_dim, act_str), act_fn])
        if continuous_actions:
            layers.extend([Linear(actor_layers[-1], np.prod(envs.single_action_space.shape), 'sigmoid'), nn.Sigmoid()])
            self.logstd = nn.Parameter(th.zeros(1, np.prod(envs.single_action_space.shape)))
            self.get_action = self.get_continuous_action
        else:
            layers.append(Linear(actor_layers[-1], np.prod(envs.single_action_space.n)))
            self.get_action = self.get_discrete_action
        self.actor = nn.Sequential(*layers)

    def get_value(self, x):
        return self.critic(x)

    def get_discrete_action(self, x, action=None, deterministic=False):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            if deterministic:
                action = th.argmax(logits, dim=-1)
            else:
                action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def get_continuous_action(self, x, action=None, deterministic=False):
        action_mu = self.actor(x)
        action_logstd = self.logstd.expand_as(action_mu)
        action_std = th.exp(action_logstd)
        probs = Normal(action_mu, action_std)
        if action is None:
            if deterministic:
                action = action_mu
            else:
                action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)