# RL2Grid Reward Function Paper - Code Setup Guide

This repository contains the code used in my research paper on **reward function design for power grid control**.
The implementation builds upon the  [RL2Grid](https://openreview.net/forum?id=7J2C4QnQrl) framework (OpenReview version).

Follow the steps below to reproduce the result of my paper.

---

## 🛠️ 1. Installation

First, install Miniconda, cd to this folder, run these to create and activate the conda environment, then install hungto lib in editable mode:

```bash
conda env create -f conda_env.yml
conda activate hungto-reward
pip install -e .
```

Then, install the required power grid simulation packages:

```bash
pip install grid2op==1.10.2
pip install lightsim2grid
pip install pandapower==2.14.11
```

## 🗂️ 2. Data Preparation

Run the notebook to split the training, validation, and testing datasets:

hungto/ExternalLibs/RL2Grid_v0/RL2Grid/tutorials/split_data.ipynb

## ⚙️ 3. Environment Generation

Before running training, modify the following file:

hungto/ExternalLibs/RL2Grid_v0/RL2Grid/env/utils.py
Inside the make_env() function, temporarily set:

generate_class = True
experimental_read_from_local_dir = False

Then run the following command to generate the environment:

```bash
python hungto/ExternalLibs/RL2Grid_v0/RL2Grid/main.py --alg PPO --total-timesteps 600000 --exp-tag YOURAGENTNAME --eval-freq 20000 --n-eval-episodes 100 --eval-env-id bus14_val --reward_fn LineCubeRootRewardNonRegularized --reward_factors 1.0 --reward_param_lsmrm_n_safe 3 --reward_param_lsmrm_n_overflow 3 --use-heuristic True --heuristic_type idle --action-type topology --additional-timesteps 100000 --n-minibatches 4 --n-envs 5 --n-steps 400 --n-threads 5 --deterministic-action False --gamma 0.9 --env-id bus14_train --vf-coef 0.5 --actor-lr 0.00003 --norm-adv True --norm-obs True --anneal-lr True --clip-coef 0.2 --critic-lr 0.0003 --difficulty 1 --gae-lambda 0.95 --clip-vfloss True --actor-act-fn tanh --actor-layers 256 128 64 --entropy-coef 0.01 --optimize-mem False --critic-act-fn tanh --critic-layers 512 256 256 --max-grad-norm 10 --update-epochs 40 --env-config-path scenario.json --th-deterministic True --wandb-mode offline --time-limit 9000 --seed 42 --cuda True --verbose True --track True
```

You will see a message confirming successful environment generation.

## 🔁 4. Restore Environment Flags and Train

Now revert the changes in make_env():

generate_class = False
experimental_read_from_local_dir = True

Then re-run the same command to begin training:

```bash
python hungto/ExternalLibs/RL2Grid_v0/RL2Grid/main.py --alg PPO --total-timesteps 600000 --exp-tag YOURAGENTNAME --eval-freq 20000 --n-eval-episodes 100 --eval-env-id bus14_val --reward_fn LineCubeRootRewardNonRegularized --reward_factors 1.0 --reward_param_lsmrm_n_safe 3 --reward_param_lsmrm_n_overflow 3 --use-heuristic True --heuristic_type idle --action-type topology --additional-timesteps 100000 --n-minibatches 4 --n-envs 5 --n-steps 400 --n-threads 5 --deterministic-action False --gamma 0.9 --env-id bus14_train --vf-coef 0.5 --actor-lr 0.00003 --norm-adv True --norm-obs True --anneal-lr True --clip-coef 0.2 --critic-lr 0.0003 --difficulty 1 --gae-lambda 0.95 --clip-vfloss True --actor-act-fn tanh --actor-layers 256 128 64 --entropy-coef 0.01 --optimize-mem False --critic-act-fn tanh --critic-layers 512 256 256 --max-grad-norm 10 --update-epochs 40 --env-config-path scenario.json --th-deterministic True --wandb-mode offline --time-limit 9000 --seed 42 --cuda True --verbose True --track True

```

The folder "runs" will appear and you can use tensorboard to visualize.

Then you can take the checkpoint file and run test_PPO.py for testing phase, I've built this to create image for the paper and the data for grid2viz. Cd to this folder and:


```bash
python   | Abs path to the test_PPO.py on your machine |   --checkpoint-path | Abs path of your checkpoint file |   --num-runner-episodes 100 --runner-output-dir "./namethatyouwant" --seed 123 --eval-env-id "bus14_val" --env-id "bus14_train" --action-type "topology" --difficulty 1 --cuda True --env-config-path "scenario.json" --norm-obs True  --use-heuristic True --th-deterministic True --n-threads 4 --deterministic-action False --alg PPO --norm-reward False --heuristic-type idle
```
Note that the result can vary between different seed, even with the same running seed, different combination of split data (train/val/test) can result to different visualization. Sometimes when I run (with train/val/test data on my local machine) the reward LineCubeRootRewardNonRegularized and LineCubeRootReward can have different performance...

Note: I need to refactor this file, too slow...


📝 Notes

If this doesn't work, read the README file of hungto/ExternalLibs/RL2Grid_v0/RL2Grid, that is the original guide to install RL2Grid.
