from hungto.ExternalLibs.RL2Grid_v0.RL2Grid.common.imports import ap
from hungto.ExternalLibs.RL2Grid_v0.RL2Grid.common.utils import str2bool

def get_env_args():
    parser = ap.ArgumentParser()

    # Settings
    parser.add_argument("--env-id", type=str, default="bus14", help="ID of the grid2op environment")
    parser.add_argument("--n-envs", type=int, default=2, help="Number of parallel envs to run")
    parser.add_argument("--action-type", type=str, default="topology", choices=["topology", "redispatch"], help="Type of environment: topology (discrete) or redispatch (continuous)")
    parser.add_argument("--difficulty", type=int, default=0, help="Higher difficulty means bigger action spaces")

    # Scenarios
    parser.add_argument("--env-config-path", type=str, default="scenario.json", help="Path to environment configuration file")

    # Normalization
    parser.add_argument("--norm-obs", type=str2bool, default=True, help="Toggle normalize observations")
    parser.add_argument("--norm-reward", type=str2bool, default=False, help="Toggle normalize rewards")
    
    # Heuristic
    parser.add_argument("--use-heuristic", type=str2bool, default=True, help="Toggles heuristics for base operations. If True, heuristic_type is used.")
    parser.add_argument("--heuristic_type", type=str, default="idle",
                        choices=["idle", "reco_revert", "idle_non_loop"],
                        help="Specify the type of heuristic to use if --use-heuristic is True.")
    

    
    # Custom Rewards via CLI
    parser.add_argument("--reward_fn", nargs='+', type=str, default=None, 
                        help="List of reward function names to use (e.g., IncreasingFlatReward LineMarginReward)")
    parser.add_argument("--reward_factors", nargs='+', type=float, default=None, 
                        help="List of corresponding reward factors (e.g., 0.1 0.9)")

    # Parameters for specific rewards (if specified in --reward_fn)

    # LineCubeRootReward & LineCubeRootRewardNonRegularized (shared for simplicity, can be split if needed)
    parser.add_argument("--reward_param_lsmrm_n_safe", type=int, default=3, help="n_th_root_safe for LineSoftMaxRootMarginReward variants")
    parser.add_argument("--reward_param_lsmrm_n_overflow", type=int, default=3, help="n_th_root_overflow for LineSoftMaxRootMarginReward variants")



    # Environment
    parser.add_argument("--eval-env-id", type=str, default="bus14_val", help="Environment ID for evaluation")
    # Evaluation parameters
    parser.add_argument("--eval-freq", type=int, default=10000, help="Frequency of evaluation in terms of environment steps")
    parser.add_argument("--n-eval-episodes", type=int, default=50, help="Number of episodes to evaluate on")

    # Parse the arguments
    params, _ = parser.parse_known_args()

    return params

