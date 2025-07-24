from hungto.ExternalLibs.RL2Grid_v0.RL2Grid.common.utils import Linear, th_act_fns
from hungto.ExternalLibs.RL2Grid_v0.RL2Grid.common.imports import nn, np, th

class QNetwork(nn.Module):
    def __init__(self, envs, args):
        super().__init__()

        act_str, act_fn = args.act_fn, th_act_fns[args.act_fn]
        layers = []
        layers.extend([
            Linear(np.array(envs.single_observation_space.shape).prod(), args.layers[0], act_str), 
            act_fn
        ])
        for idx, embed_dim in enumerate(args.layers[1:], start=1): 
            layers.extend([Linear(args.layers[idx-1], embed_dim, act_str), act_fn])
        layers.append(Linear(args.layers[-1], envs.single_action_space.n, 'linear'))
        self.qnet = nn.Sequential(*layers)

    def forward(self, x):
        return self.qnet(x)
    
    def get_action(self, x):
        q_values = self(x)
        actions = th.argmax(q_values, dim=1).cpu().numpy()
        return actions