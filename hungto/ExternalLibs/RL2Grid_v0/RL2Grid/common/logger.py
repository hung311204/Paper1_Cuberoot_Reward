import os
import subprocess
from collections import deque 

import shutil
from torch.utils.tensorboard import SummaryWriter

from .imports import wb, np

class Logger:
    def __init__(self, run_name, args, log_freq=30):
        self.log_freq = log_freq

        self.episodic_survival = deque(maxlen=log_freq)
        self.episodic_return = deque(maxlen=log_freq)
        self.episodic_length = deque(maxlen=log_freq)

        self.wb_mode = args.wandb_mode

        wb_path = wb.init(
            name=run_name,
            id=run_name,
            config=vars(args),
            mode=self.wb_mode,
            project=args.wandb_project,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            settings=wb.Settings(_disable_stats=True),
            resume=True
        )
        self.wb_path = os.path.split(wb_path.dir)[0]

        
        self.writer = SummaryWriter(f"runs/{run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        
    def store_metrics(self, survival, info):
        self.episodic_survival.append(survival)
        self.episodic_return.append(info['episode']['r'][0])
        self.episodic_length.append(info['episode']['l'][0])

    def log_metrics(self, global_step):
        record = {
            "charts/episodic_survival": np.mean(self.episodic_survival),
            "charts/episodic_return": np.mean(self.episodic_return),
            "charts/episodic_length": np.mean(self.episodic_length),
            "charts/global_step": global_step
        }
        wb.log(record, step=global_step)
        
        self.writer.add_scalar("charts/episodic_survival", np.mean(self.episodic_survival), global_step)
        self.writer.add_scalar("charts/episodic_return", np.mean(self.episodic_return), global_step)
        self.writer.add_scalar("charts/episodic_length", np.mean(self.episodic_length), global_step)
        
        #self.episodic_survival.clear()
        #self.episodic_return.clear()
        #self.episodic_length.clear()

    def close(self):
        #self.writer.close()
        if self.wb_path is not None and self.wb_mode == 'offline':
            wb.finish()
            subprocess.run(['wandb', 'sync', self.wb_path]) 
            # shutil.rmtree(self.wb_path)   # Remove wandb run folder
            
            ''' 
            if result.returncode == 0:  # Only remove if sync successful
                shutil.rmtree(self.wb_path)
            else:
                print("Warning: wandb sync failed. Data remains in:", self.wb_path)

            '''



