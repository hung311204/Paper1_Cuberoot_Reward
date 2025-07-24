import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .imports import th

@dataclass
class CheckpointSaver(ABC):
    run_name: str
    args: dict

    def __post_init__(self):
        self.ckpt_dir = 'checkpoint'
        if not os.path.exists(self.ckpt_dir): os.makedirs(self.ckpt_dir)
        self.loaded_run, self.record = {}, {}
        if self.args.resume_run_name:
            checkpoint_name = 'checkpoint/' + self.args.resume_run_name + '.tar'
            self.loaded_run = th.load(checkpoint_name)
            #os.remove(checkpoint_name)

    @property
    def resumed(self):
        return self.loaded_run != {}
    
    def _get_base_record(self, global_step):
        """
        Save a copy of state_dict for each network and optimizer.
        """

        # We also want to save all the kwargs since we have to resume a run (i.e., alg and hyperparams) while ignoring the current settings in the config files
        self.record = {
            'global_step': global_step,
        }

    def save(self):
        th.save(
            self.record, self.ckpt_dir + '/' + self.run_name + '.tar'
        )

    @abstractmethod
    def set_record():
        pass
        
class PPOCheckpoint(CheckpointSaver):
    def set_record(self, args, actor, critic, global_step, actor_optim, critic_optim, wb_run_name, last_rollout=0):
        self._get_base_record(global_step)
        self.record['args'] = args
        self.record['actor'] = actor.state_dict()
        self.record['critic'] = critic.state_dict()
        self.record['actor_optim'] = actor_optim.state_dict()
        self.record['critic_optim'] = critic_optim.state_dict()
        self.record['wb_run_name'] = wb_run_name
        self.record['last_rollout'] = last_rollout

class SACCheckpoint(PPOCheckpoint):
    def set_record(self, args, alpha, actor, critic, critic2, global_step, actor_optim, critic_optim, wb_run_name, last_step=0):
        super().set_record(args, actor, critic, global_step, actor_optim, critic_optim, wb_run_name)
        self.record['critic2'] = critic2.state_dict()
        self.record['alpha'] = alpha
        self.record['last_step'] = last_step

class TD3Checkpoint(PPOCheckpoint):
    def set_record(self, args, actor, critic, critic2, global_step, actor_optim, critic_optim, wb_run_name, last_step=0):
        super().set_record(args, actor, critic, global_step, actor_optim, critic_optim, wb_run_name)
        self.record['critic2'] = critic2.state_dict()
        self.record['last_step'] = last_step

class DQNCheckpoint(CheckpointSaver):
    def set_record(self, args, qnet, global_step, qnet_optim, wb_run_name, last_step=0):
        self._get_base_record(global_step)
        self.record['args'] = args
        self.record['qnet'] = qnet.state_dict()
        self.record['qnet_optim'] = qnet_optim.state_dict()
        self.record['wb_run_name'] = wb_run_name
        self.record['last_step'] = last_step
