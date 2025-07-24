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

    agent.eval() # Ensure the Agent module (and its submodules like self.actor) are in eval mode
    return agent, checkpoint['args'] # Return the whole agent instance

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
        self.use_heuristic = use_heuristic # Flag to enable/disable heuristic logic
        self.actor_network.eval() # Ensure model is in evaluation mode

    def act(self, observation, reward, done):
        return self.action_space({}) # Always return a do nothing action 


if __name__ == "__main__":
    parser = ap.ArgumentParser()

    # Checkpoint
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to the checkpoint file")

    # Environment
    parser.add_argument("--env-id", type=str, default="bus14", help="Environment ID the model was trained on (e.g., bus14, bus14_train)")
    parser.add_argument("--difficulty", type=int, default=0, help="Difficulty level of the environment (used for creating eval env if not in loaded_args)")
    parser.add_argument("--eval-env-id", type=str, default="bus14_val",
                        help="The ID of the environment to use for evaluation (e.g., bus14_test, bus14_val). If None, uses --env-id.")
    parser.add_argument("--action-type", type=str, default="topology", help="Type of action space (e.g., topology, redispatch)")
    parser.add_argument("--env-config-path", type=str, default="scenario.json", help="Path to the environment configuration file")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed for evaluation environment")

    # Torch
    parser.add_argument("--cuda", type=str2bool, default=False, help="Enable CUDA by default.")
    parser.add_argument("--n-threads", type=int, default=8, help="Max number of torch threads.")

    # Runner specific
    parser.add_argument("--runner-output-dir", type=str, default="grid2op_runner_results_ppo", help="Directory to save Grid2Op Runner outputs.")
    parser.add_argument("--num-runner-episodes", type=int, default=5, help="Number of episodes for Grid2Op Runner to evaluate.")

    # Environment parameters (simplified for testing, copy relevant levels from main_*.py)
    # These are used if you want to apply specific Grid2Op Parameters during evaluation
    # For simplicity, we'll use args.difficulty to pick a level.
    # You might want to make this more sophisticated or load from a shared config.
    # Note: This should ideally match the ENV_PARAMS used during training if you want
    # to evaluate on the same difficulty settings.
    global ENV_PARAMS_FOR_TEST # Make it global to be accessible for param creation
    ENV_PARAMS_FOR_TEST = {
        0: {}, # Default, no specific overrides beyond MAX_LINE/SUB_CHANGED
        1: {"HARD_OVERFLOW_THRESHOLD": 999, "NB_TIMESTEP_OVERFLOW_ALLOWED": 9999999, "SOFT_OVERFLOW_THRESHOLD": 1.0, "NO_OVERFLOW_DISCONNECTION": True},
        2: {"HARD_OVERFLOW_THRESHOLD": 999, "NB_TIMESTEP_OVERFLOW_ALLOWED": 30, "SOFT_OVERFLOW_THRESHOLD": 1.0},
        3: {"HARD_OVERFLOW_THRESHOLD": 999, "NB_TIMESTEP_OVERFLOW_ALLOWED": 20, "SOFT_OVERFLOW_THRESHOLD": 1.0},
        4: {"HARD_OVERFLOW_THRESHOLD": 999, "NB_TIMESTEP_OVERFLOW_ALLOWED": 10, "SOFT_OVERFLOW_THRESHOLD": 1.0},
        5: {"HARD_OVERFLOW_THRESHOLD": 2.0, "NB_TIMESTEP_OVERFLOW_ALLOWED": 3, "SOFT_OVERFLOW_THRESHOLD": 1.0}
    }
    parser.add_argument("--norm-obs", type=str2bool, default=False, help="Normalize observation")
    parser.add_argument("--use-heuristic", type=str2bool, default=True, help="Use heuristic")

    cmd_line_args = parser.parse_args()

    # Load the algorithm-specific arguments (defaults for PPO if not in checkpoint)
    alg_args = get_alg_args()
    # Combine command line args with algorithm defaults
    args = ap.Namespace(**vars(cmd_line_args), **vars(alg_args))

    eval_seed_from_cmd = args.seed # Capture seed specifically for evaluation

    # Determine the environment ID for evaluation
    eval_env_id_to_use = args.eval_env_id if args.eval_env_id else args.env_id
    print(f"Setting up evaluation environment: {eval_env_id_to_use}")

    # --- Prepare arguments and parameters for environment creation ---
    env_creation_args = ap.Namespace(**vars(args)) # Start with a copy of current args
    env_creation_args.env_id = eval_env_id_to_use
    # env_creation_args.seed will be eval_seed_from_cmd (set below)

    grid_eval_params = Parameters()
    grid_eval_params.MAX_LINE_STATUS_CHANGED = 1 # Default from main_*.py
    grid_eval_params.MAX_SUB_CHANGED = 1       # Default from main_*.py

    param_level_for_eval = args.difficulty
    if param_level_for_eval in ENV_PARAMS_FOR_TEST:
        params_dict = ENV_PARAMS_FOR_TEST[param_level_for_eval]
        if params_dict: # Apply only if there are params for this level
            print(f"Runner: Using Grid2Op parameter set level {param_level_for_eval} for eval environment:")
            for key, value in params_dict.items():
                print(f"  {key}: {value}")
                setattr(grid_eval_params, key, value)
    else:
        print(f"Runner: Warning: Invalid param level {param_level_for_eval} for Grid2Op Parameters. Using defaults + MAX_LINE/SUB_CHANGED=1.")

    # --- Create a Gym-wrapped environment to get converters and for Actor init ---
    # This GymEnv will use the eval_env_id and grid_eval_params
    # We use a single environment (idx=0) and the eval_seed
    env_creation_args.seed = eval_seed_from_cmd # Ensure make_env uses the eval seed
    temp_gym_env_thunk = make_env(env_creation_args, idx=0, resume_run=False, params=grid_eval_params)
    gym_env_for_eval_config = temp_gym_env_thunk()

    # This AsyncVectorEnv is primarily for Actor initialization if it expects a VectorEnv interface
    # It's based on the evaluation environment's configuration.
    vec_env_for_actor_init = gym.vector.AsyncVectorEnv([
        lambda: make_env(env_creation_args, i, resume_run=False, params=grid_eval_params)() for i in range(1)
    ])

    set_random_seed(eval_seed_from_cmd) # Use eval_seed for global random state during testing
    set_torch(args.n_threads, True, args.cuda)
    device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")

    # Load the checkpoint
    # Actor will be initialized using the observation/action spaces from vec_env_for_actor_init
    loaded_agent_instance, loaded_checkpoint_args = load_checkpoint(args.checkpoint_path, vec_env_for_actor_init, args)
    loaded_agent_instance.to(device) # Move the whole agent to device

    print(f"Successfully loaded checkpoint. Model was trained with env_id: {loaded_checkpoint_args.env_id} and seed: {loaded_checkpoint_args.seed}")
    print(f"Evaluating on env_id: {eval_env_id_to_use} with seed: {eval_seed_from_cmd}")

    # --- Setup for Grid2Op Runner ---
    # Get the raw Grid2Op environment from the Gym wrapper's init_env attribute
    g2op_eval_env = gym_env_for_eval_config.init_env
    # Instantiate the agent wrapper
    agent_wrapper = PPOAgentWrapper(
        actor_network=loaded_agent_instance, # Pass the whole Agent instance (which has get_action)
        g2op_env_action_space=g2op_eval_env.action_space, # Use the action space from the raw g2op env for BaseAgent init (needed for get_do_nothing_action)
        gym_obs_converter=gym_env_for_eval_config.observation_space, # Use the Gym wrapper's observation space converter
        gym_act_converter=gym_env_for_eval_config.action_space,   # Use the Gym wrapper's action space converter
        device=device,
        use_heuristic=args.use_heuristic # Pass the use_heuristic flag from command line args
    )

    # Initialize Runner as per the requested pattern
    # The Runner needs the environment instance that includes all desired wrappers (like heuristic)
    # Pass parameters from the raw Grid2Op environment, as Runner works with raw envs
    runner = Runner(**g2op_eval_env.get_params_for_runner(),
                    agentClass=None, # We provide an instance
                    agentInstance=agent_wrapper)

    print(f"\nStarting Grid2Op Runner evaluation for {args.num_runner_episodes} episodes...")

    # Call runner.run() with parameters as per the requested pattern
    results_summary = runner.run(nb_episode=args.num_runner_episodes,
                                 max_iter=-1,  # -1 means run episodes to their natural end
                                 pbar=tqdm, # Pass the tqdm class for progress bar
                                 path_save=args.runner_output_dir,
                                 add_detailed_output = True)

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
        max_timesteps = [res[4] for res in results_summary] # Assuming max_timesteps can vary per chronic
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
    vec_env_for_actor_init.close()
    # g2op_eval_env is owned by gym_env_for_eval_config, so it should be closed by its wrapper.
    
    