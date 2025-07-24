import numpy as np
import torch as th
from hungto.ExternalLibs.RL2Grid_v0.RL2Grid.common.imports import th, np
from hungto.ExternalLibs.RL2Grid_v0.RL2Grid.common.logger import Logger

class PPOMetrics:
    def __init__(self):
        self.returns = []
        self.lengths = []
        self.survival_rates = []
        
    def update(self, episode_return, episode_length, max_steps):
        self.returns.append(episode_return)
        self.lengths.append(episode_length)
        self.survival_rates.append(episode_length / max_steps)
        
    def get_metrics(self):
        return {
            "eval/mean_return": np.mean(self.returns),
            "eval/std_return": np.std(self.returns),
            "eval/mean_length": np.mean(self.lengths),
            "eval/mean_survival_rate": np.mean(self.survival_rates)
        }
    
    def reset(self):
        self.returns = []
        self.lengths = []
        self.survival_rates = []

def evaluate_policy(agent, eval_env, n_episodes, max_steps, device):
    """Evaluate the policy for n_episodes and return metrics"""
    metrics = PPOMetrics()
    
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        
        while not done:
            obs = th.tensor(obs).to(device)
            with th.no_grad():
                # Use deterministic policy (mean of normal distribution)
                action, _, _ = agent.get_action(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = eval_env.step(action.cpu().numpy())
            done = terminated or truncated
            
            if done:
                if "episode" in info:
                    episode_length = info['episode']['l'][0]
                    episode_return = info['episode']['r'][0]
                    metrics.update(episode_return, episode_length, max_steps)
            
            if info.get('episode', {}).get('l', [0])[0] >= max_steps:
                break
                
    return metrics.get_metrics() 