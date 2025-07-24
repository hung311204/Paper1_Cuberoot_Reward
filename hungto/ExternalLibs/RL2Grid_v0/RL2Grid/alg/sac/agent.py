from torch.distributions import Normal, Categorical

from hungto.ExternalLibs.RL2Grid_v0.RL2Grid.common.utils import Linear, th_act_fns
from hungto.ExternalLibs.RL2Grid_v0.RL2Grid.common.imports import F, nn, np, th

# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, envs, args, continuous_actions):
        super().__init__()

        critic_layers = args.critic_layers
        act_str, self.act_fn = args.critic_act_fn, th_act_fns[args.critic_act_fn]

        self.fc_i = Linear(
            np.prod(envs.single_observation_space.shape) + np.prod(envs.single_action_space.shape) if continuous_actions else np.prod(envs.single_observation_space.shape),      
            critic_layers[0], 
            act_str
        )
        
        self.fc_h = []
        for idx, embed_dim in enumerate(critic_layers[1:], start=1): 
            self.fc_h.append(Linear(critic_layers[idx-1], embed_dim, act_str))

        self.forward = self.forward_discrete
        if continuous_actions: 
            self.fc_o = Linear(critic_layers[-1], 1, 'linear')
            self.forward = self.forward_continuous

        else: self.fc_o = Linear(critic_layers[-1], envs.single_action_space.n, 'linear')

    def forward_continuous(self, x, a):
        x = self.act_fn(self.fc_i(th.cat([x, a], 1)))
        for fc in self.fc_h: x = self.act_fn(fc(x))
        return self.fc_o(x)
    
    def forward_discrete(self, x):
        x = self.act_fn(self.fc_i(x))
        for fc in self.fc_h: x = self.act_fn(fc(x))
        return self.fc_o(x)     

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self, envs, args, continuous_actions):
        super().__init__()

        actor_layers = args.actor_layers
        act_str, self.act_fn = args.actor_act_fn, th_act_fns[args.actor_act_fn]
        self.fc_i = Linear(
            np.prod(envs.single_observation_space.shape), actor_layers[0], 
            act_str
        )

        self.fc_h = []
        for idx, embed_dim in enumerate(actor_layers[1:], start=1): 
            self.fc_h.append(Linear(actor_layers[idx-1], embed_dim, act_str))

        if continuous_actions:
            self.fc_mu = Linear(actor_layers[idx], np.prod(envs.single_action_space.shape), 'linear')
            self.fc_logstd = Linear(actor_layers[idx], np.prod(envs.single_action_space.shape), 'linear')

            # action rescaling
            self.register_buffer(
                "action_scale", th.tensor((envs.single_action_space.high - envs.single_action_space.low) / 2.0, dtype=th.float32)
            )
            self.register_buffer(
                "action_bias", th.tensor((envs.single_action_space.high + envs.single_action_space.low) / 2.0, dtype=th.float32)
            )
            self.forward = self.continuous_forward
            self.get_action = self.get_continuous_action
        else:
            self.fc_o = Linear(actor_layers[-1], np.prod(envs.single_action_space.n), 'linear')
            self.forward = self.discrete_forward
            self.get_action = self.get_discrete_action

    def continuous_forward(self, x):
        x = self.act_fn(self.fc_i(x))
        for fc in self.fc_h: x = self.act_fn(fc(x))
        mean = self.fc_mu(x)
        log_std = self.fc_logstd(x)
        log_std = th.tanh_(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_continuous_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = th.tanh_(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= th.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = th.tanh_(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    
    def discrete_forward(self, x):
        x = self.act_fn(self.fc_i(x))
        for fc in self.fc_h: x = self.act_fn(fc(x))
        return self.fc_o(x)

    def get_discrete_action(self, x):
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs