
#import multiprocessing as mp
#mp.set_start_method('spawn', force=True)

# TODO here it use the new arg (from cli, not from ckpt), that's not good, should use old arg which is loaded from ckpt

import time
import numpy as np
import torch as th
import gymnasium as gym
from env.utils import make_env, DiscreteActSpace # To create the environment and import DiscreteActSpace (needed for isinstance check)
from alg.ppo.agent import Agent # To load the PPO Agent architecture (contains Actor)
from common.checkpoint import PPOCheckpoint #to load the checkpoint
from common.utils import set_random_seed, set_torch, str2bool
from common.imports import ap
from env.config import get_env_args
from alg.ppo.config import get_alg_args #import here

from grid2op import make as g2op_make
from grid2op.Agent import BaseAgent
from grid2op.Runner import Runner
from grid2op.Parameters import Parameters
from env.heuristic import RHO_SAFETY_THRESHOLD # Import the safety threshold
from tqdm import tqdm # Import tqdm for the progress bar
import matplotlib.pyplot as plt
import os
import json
import pandas as pd  # For saving tabular observations to CSV


def load_checkpoint(checkpoint_path, envs, args):
    """Loads the PPO checkpoint from the given path."""
    checkpoint = th.load(checkpoint_path)

    # Create an Agent instance (which contains the Actor and Critic)
    # The Agent needs the environment spaces and args to build the network architecture
    # We also need to know if actions are continuous or discrete for the Agent constructor
    continuous_actions = True if args.action_type == "redispatch" else False
    agent = Agent(envs, args, continuous_actions)

    # Load the state dictionary into the agent's actor network (which is an nn.Sequential)
    # The checkpoint['actor'] contains the state_dict for the self.actor (nn.Sequential) part of the Agent class
    agent.actor.load_state_dict(checkpoint['actor'])
    # agent.critic.load_state_dict(checkpoint['critic']) # Critic is not needed for inference

    agent.eval()  # Ensure the Agent module (and its submodules like self.actor) are in eval mode
    return agent, checkpoint['args']  # Return the whole agent instance


class PPOAgentWrapper(BaseAgent):
    """
    A wrapper for the trained PPO Actor network to make it compatible with grid2op.Runner.
    """
    def __init__(self, actor_network, g2op_env_action_space,
                 gym_obs_converter, gym_act_converter, device, use_heuristic):
        """
        Args:
            actor_network: The trained PyTorch PPO Agent instance (which has the get_action method).
            g2op_env_action_space: The native Grid2Op action space from the evaluation environment.
                                   (Used for BaseAgent initialization and getting do_nothing_action).
                                   NOTE: This should be the action space of the *raw* grid2op env.
            gym_obs_converter: The Gymnasium observation space wrapper (e.g., BoxGymObsSpace)
                               used to convert Grid2Op observations to NumPy arrays.
            gym_act_converter: The Gymnasium action space wrapper (e.g., DiscreteActSpace or BoxGymActSpace)
                               used to convert action indices to Grid2Op actions.
            device: The torch device (e.g., "cpu", "cuda").
        """
        super().__init__(g2op_env_action_space)
        self.actor_network = actor_network
        self.gym_obs_converter = gym_obs_converter
        self.gym_act_converter = gym_act_converter
        self.device = device
        self.use_heuristic = use_heuristic  # Flag to enable/disable heuristic logic
        self.actor_network.eval()  # Ensure model is in evaluation mode

    def act(self, observation, reward, done):
        # Heuristic Logic: If use_heuristic is True and the grid is safe, return do-nothing
        if self.use_heuristic and observation.rho.max() < RHO_SAFETY_THRESHOLD:
            return self.action_space({})

        # Otherwise, use the RL agent's policy
        # Convert Grid2Op observation to Gym observation array
        gym_obs_array = self.gym_obs_converter.to_gym(observation)
        # Convert Gym observation array to PyTorch tensor, add batch dimension
        obs_tensor = th.tensor(gym_obs_array, dtype=th.float32).unsqueeze(0).to(self.device)

        with th.no_grad():
            # Get action from the actor network
            # The Actor's get_action method returns action, logprob, entropy
            # self.actor_network is now the Agent instance, which has get_action
            # (which internally calls get_discrete_action or get_continuous_action)
            action, _, _ = self.actor_network.get_action(obs_tensor, deterministic=True)

            # For discrete actions, the action is an index (tensor). For continuous, it's a tensor of values.
            # The gym_act_converter.from_gym expects the Gym action format (int for discrete, array for box)
            if isinstance(self.gym_act_converter, DiscreteActSpace):
                action_idx = action.cpu().item()  # Get Python number for discrete

        # Convert the action index to a Grid2Op action object
        grid2op_action = self.gym_act_converter.from_gym(action_idx)
        return grid2op_action


if __name__ == "__main__":
    # --- Argument Parsing using get_env_args and get_alg_args ---
    parser = ap.ArgumentParser()
    # Add only minimal custom arguments, let get_env_args/get_alg_args handle the rest
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--runner-output-dir", type=str, default="grid2op_runner_results_ppo", help="Directory to save Grid2Op Runner outputs.")
    parser.add_argument("--num-runner-episodes", type=int, default=5, help="Number of episodes for Grid2Op Runner to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--th-deterministic", type=str2bool, default=True, help="Enable deterministic in Torch.")
    parser.add_argument("--cuda", type=str2bool, default=False, help="Enable CUDA by default.")
    parser.add_argument("--n-threads", type=int, default=8, help="Max number of torch threads.")
    parser.add_argument("--deterministic-action", type=str2bool, default=False, help="Use deterministic actions during training.")
    # Add new argument for testing environment id
    parser.add_argument("--env-testing-id", type=str, default="bus14_test", help="Environment ID for testing (overrides eval_env_id/env_id)")

    # Parse known args first
    args, _ = parser.parse_known_args()
    # Merge with defaults from get_env_args and get_alg_args
    args = ap.Namespace(**vars(args), **vars(get_env_args()), **vars(get_alg_args()))

    # --- Determine which environment id to use for testing ---
    env_testing_id = args.env_testing_id or args.eval_env_id or args.env_id
    print(f"Setting up testing environment: {env_testing_id}")

    # --- Prepare environment creation arguments (from CLI/config) ---
    env_creation_args = ap.Namespace(**vars(args))
    env_creation_args.env_id = env_testing_id

    # --- Grid2Op Parameters ---
    param_level_for_eval = getattr(args, "difficulty", 0)
    ENV_PARAMS_FOR_TEST = {
        0: {},
        1: {"HARD_OVERFLOW_THRESHOLD": 2.0, "NB_TIMESTEP_OVERFLOW_ALLOWED": 3, "SOFT_OVERFLOW_THRESHOLD": 1.0},
        2: {"HARD_OVERFLOW_THRESHOLD": 999, "NB_TIMESTEP_OVERFLOW_ALLOWED": 30, "SOFT_OVERFLOW_THRESHOLD": 1.0},
        3: {"HARD_OVERFLOW_THRESHOLD": 999, "NB_TIMESTEP_OVERFLOW_ALLOWED": 20, "SOFT_OVERFLOW_THRESHOLD": 1.0},
        4: {"HARD_OVERFLOW_THRESHOLD": 999, "NB_TIMESTEP_OVERFLOW_ALLOWED": 10, "SOFT_OVERFLOW_THRESHOLD": 1.0},
        5: {"HARD_OVERFLOW_THRESHOLD": 2.0, "NB_TIMESTEP_OVERFLOW_ALLOWED": 3, "SOFT_OVERFLOW_THRESHOLD": 1.0}
    }
    grid_eval_params = Parameters()
    grid_eval_params.MAX_LINE_STATUS_CHANGED = 1
    grid_eval_params.MAX_SUB_CHANGED = 1
    if param_level_for_eval in ENV_PARAMS_FOR_TEST:
        params_dict = ENV_PARAMS_FOR_TEST[param_level_for_eval]
        if params_dict:  # Apply only if there are params for this level
            print(f"Runner: Using Grid2Op parameter set level {param_level_for_eval} for eval environment:")
            for key, value in params_dict.items():
                print(f"  {key}: {value}")
                setattr(grid_eval_params, key, value)
    else:
        print(f"Runner: Warning: Invalid param level {param_level_for_eval} for Grid2Op Parameters. Using defaults + MAX_LINE/SUB_CHANGED=1.")

    # --- Step 1: Create a minimal environment for loading the checkpoint (spaces only) ---
    vec_env_for_actor_init = gym.vector.AsyncVectorEnv([
        lambda: make_env(env_creation_args, i, resume_run=False, params=grid_eval_params)() for i in range(1)
    ])

    # --- Step 2: Load the checkpoint ---
    # Ensure device is defined before using it
    if 'device' not in locals():
        device = th.device("cuda" if args.cuda and th.cuda.is_available() else "cpu")
    loaded_agent_instance, loaded_checkpoint_args = load_checkpoint(args.checkpoint_path, vec_env_for_actor_init, args)
    loaded_agent_instance.to(device)  # Move the whole agent to device

    # --- Step 3: Merge CLI/config args with checkpoint args (CLI/config takes precedence) ---
    cli_args_dict = vars(args)
    ckpt_args_dict = vars(loaded_checkpoint_args)
    merged_args_dict = {**ckpt_args_dict, **cli_args_dict}
    args = ap.Namespace(**merged_args_dict)

    # --- Step 4: Re-create the environment using the merged args (with env_id set to the testing id) ---
    env_creation_args = ap.Namespace(**vars(args))
    env_creation_args.env_id = env_testing_id

    temp_gym_env_thunk = make_env(env_creation_args, idx=0, resume_run=False, params=grid_eval_params, eval_mode=True)
    gym_env_for_eval_config = temp_gym_env_thunk()

    # Clean up the temp env used for loading the checkpoint
    vec_env_for_actor_init.close()

    set_random_seed(args.seed)
    set_torch(args.n_threads, True, args.cuda)
    # device = th.device("cuda" if th.cuda and th.cuda.is_available() else "cpu") # This line is now redundant

    print(f"Successfully loaded checkpoint. Model was trained with env_id: {loaded_checkpoint_args.env_id} and seed: {loaded_checkpoint_args.seed}")
    print(f"Evaluating on env_id: {env_testing_id} with seed: {args.seed}")

    # --- Setup for Grid2Op Runner ---
    # Get the raw Grid2Op environment from the Gym wrapper's init_env attribute
    g2op_eval_env = gym_env_for_eval_config.init_env
    # Instantiate the agent wrapper
    agent_wrapper = PPOAgentWrapper(
        actor_network=loaded_agent_instance,  # Pass the whole Agent instance (which has get_action)
        g2op_env_action_space=g2op_eval_env.action_space,  # Use the action space from the raw g2op env for BaseAgent init (needed for get_do_nothing_action)
        gym_obs_converter=gym_env_for_eval_config.observation_space,  # Use the Gym wrapper's observation space converter
        gym_act_converter=gym_env_for_eval_config.action_space,  # Use the Gym wrapper's action space converter
        device=device,
        use_heuristic=args.use_heuristic  # Pass the use_heuristic flag from command line args
    )

    # --- Custom Runner Loop for Survival Rate Plotting ---
    n_episode = args.num_runner_episodes
    episode_survival_steps = []
    episode_max_steps = []

    # Get max_steps from the underlying Grid2Op env (see main.py context)
    # Ensure max_steps is an integer, not a method
    max_steps = gym_env_for_eval_config.init_env.chronics_handler.max_episode_duration()
  

    for ep in range(n_episode):
        obs, _ = gym_env_for_eval_config.reset()
        done = False
        ep_survival_steps = None  # Will be set from info['episode']['l'][0]
        while not done:
            # Use the agent to select an action
            obs_tensor = th.tensor(obs, dtype=th.float32).unsqueeze(0).to(device)
            with th.no_grad():
                action, _, _ = loaded_agent_instance.get_action(obs_tensor, deterministic=True)
                if isinstance(gym_env_for_eval_config.action_space, DiscreteActSpace):
                    action_idx = action.cpu().item()
                else:
                    action_idx = action.cpu().numpy()
            obs_next, reward, done, _, info = gym_env_for_eval_config.step(action_idx)

            obs = obs_next

            # If episode ended, try to get survival steps from info
            if done and info is not None and "episode" in info and "l" in info["episode"]:
                ep_survival_steps = info["episode"]["l"][0]

        # After episode ends, record survival steps and max_steps
        # Survival steps: number of steps survived in this episode (from info)
        # Max steps: from env
        if ep_survival_steps is None:
            # Fallback: if info did not provide, use max_steps (should not happen in grid2op)
            ep_survival_steps = max_steps
        episode_survival_steps.append(ep_survival_steps)
        episode_max_steps.append(max_steps)

    # --- Plot survival rate per episode as a bar chart ---
    import matplotlib.pyplot as plt
    import numpy as np

    # Ensure all max_steps are integers, not methods
    episode_max_steps_int = []
    for ms in episode_max_steps:
        if callable(ms):
            episode_max_steps_int.append(ms())
        else:
            episode_max_steps_int.append(ms)

    survival_rates = [surv / max_s if max_s != 0 else 0.0 for surv, max_s in zip(episode_survival_steps, episode_max_steps_int)]
    episode_indices = np.arange(1, n_episode + 1)

    avg_survival_rate = np.mean(survival_rates)

    plt.figure(figsize=(10, 5))
    plt.bar(episode_indices, survival_rates, color='skyblue', label='Episode Survival Rate')
    # Add a dashed line for the average survival rate
    plt.axhline(avg_survival_rate, color='red', linestyle='--', linewidth=2, label=f'Average ({avg_survival_rate:.2f})')
    plt.xlabel("Episode")
    plt.ylabel("Survival Rate")
    plt.title("Survival Rate per Episode (Survival Steps / Max Steps)")
    plt.ylim(0, 1.05)
    plt.xticks(episode_indices)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    survival_plot_path = os.path.join(args.runner_output_dir, "survival_rate_per_episode.png")
    plt.savefig(survival_plot_path)
    plt.close()
    print(f"Saved survival rate bar chart to {survival_plot_path}")

    # --- Optionally, you can still run the original runner for summary and plots if needed ---
    # Initialize Runner as per the requested pattern
    # The Runner needs the environment instance that includes all desired wrappers (like heuristic)
    # Pass parameters from the raw Grid2Op environment, as Runner works with raw envs
    runner = Runner(**g2op_eval_env.get_params_for_runner(),
                    agentClass=None,  # We provide an instance
                    agentInstance=agent_wrapper)

    print(f"\nStarting Grid2Op Runner evaluation for {args.num_runner_episodes} episodes...")

    # Call runner.run() with parameters as per the requested pattern
    results_summary = runner.run(nb_episode=args.num_runner_episodes,   # TODO nb_episode=args.num_runner_episodes
                                 max_iter=-1,  # -1 means run episodes to their natural end
                                 pbar=tqdm,  # Pass the tqdm class for progress bar
                                 path_save=args.runner_output_dir,
                                 add_detailed_output=True)

    print("Grid2Op Runner evaluation finished.")
    print(f"Results summary: {results_summary}")
    print(f"Detailed logs and results saved in: {args.runner_output_dir}")

    # --- Plotting Results ---

    if results_summary:
        # Ensure the output directory exists
        os.makedirs(args.runner_output_dir, exist_ok=True)

        # 1. Save results_summary to a JSON file
        # EpisodeData objects are not directly JSON serializable, so we'll store a string representation or exclude them.
        # For simplicity, let's store the first 5 elements of each tuple.
        serializable_summary = []
        for res_tuple in results_summary:
            serializable_summary.append({
                "chronic_path": res_tuple[0],
                "chronic_id": res_tuple[1],
                "cumulative_reward": res_tuple[2],
                "timesteps_survived": res_tuple[3],
                "max_timesteps": res_tuple[4]
                # res_tuple[5] is EpisodeData, which we are omitting for JSON
            })
        summary_file_path = os.path.join(args.runner_output_dir, "results_summary.json")
        with open(summary_file_path, 'w') as f:
            json.dump(serializable_summary, f, indent=4)
        print(f"Saved results summary to: {summary_file_path}")

        # Apply a style for dark gray background
        plt.style.use('dark_background')

        episode_numbers = np.arange(1, len(results_summary) + 1)
        cumulative_rewards = [res[2] for res in results_summary]
        timesteps_survived = [res[3] for res in results_summary]
        max_timesteps = [res[4] for res in results_summary]  # Assuming max_timesteps can vary per chronic
        survival_rates = [(ts / mt * 100) if mt > 0 else 0 for ts, mt in zip(timesteps_survived, max_timesteps)]

        # Ensure the output directory exists
        # os.makedirs(args.runner_output_dir, exist_ok=True) # Already done above

        # 2. Bar chart for cumulative rewards
        plt.figure(figsize=(10, 6))
        plt.bar(episode_numbers, cumulative_rewards, color='deepskyblue') # Changed color for better contrast on gray
        plt.xlabel("Episode Number")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Reward per Episode")
        plt.xticks(episode_numbers)
        plt.grid(axis='y', linestyle='--')
        plot_path_rewards = os.path.join(args.runner_output_dir, "episode_rewards.png")
        plt.savefig(plot_path_rewards)
        print(f"Saved episode rewards plot to: {plot_path_rewards}")
        plt.close()

        # 3. Line plot for survival rates
        plt.figure(figsize=(10, 6))
        plt.plot(episode_numbers, survival_rates, marker='o', linestyle='-', color='coral')
        
        # Add average survival rate line
        if survival_rates:
            avg_survival_rate = np.mean(survival_rates)
            plt.axhline(avg_survival_rate, color='lightgreen', linestyle='--', label=f'Avg Survival: {avg_survival_rate:.2f}%')
            plt.legend()

        plt.xlabel("Episode Number")
        plt.ylabel("Survival Rate (%)")
        plt.title("Survival Rate per Episode")
        plt.xticks(episode_numbers)
        plt.ylim(0, 100)
        # plt.grid(True, linestyle='--') # grid is handled by seaborn style
        plot_path_survival = os.path.join(args.runner_output_dir, "survival_rates.png")
        plt.savefig(plot_path_survival)
        print(f"Saved survival rates plot to: {plot_path_survival}")
        plt.close()

        # 4. Histogram for episode durations (timesteps survived)
        plt.figure(figsize=(10, 6))
        plt.hist(timesteps_survived, bins=min(len(episode_numbers), 20), color='mediumseagreen', edgecolor='black') # Changed color
        plt.xlabel("Episode Duration (Timesteps Survived)")
        plt.ylabel("Frequency")
        plt.title("Histogram of Episode Durations")
        plt.grid(axis='y', linestyle='--')
        plot_path_durations = os.path.join(args.runner_output_dir, "episode_durations_histogram.png")
        plt.savefig(plot_path_durations)
        print(f"Saved episode durations histogram to: {plot_path_durations}")
        plt.close()

    # Clean up
    gym_env_for_eval_config.close()
    # g2op_eval_env is owned by gym_env_for_eval_config, so it should be closed by its wrapper.        
