import os
#os.environ["grid2op_class_in_file"] = "true"  # or "false" if you want to disable it
import json

#from gym.wrappers import NormalizeObservation as NormalizeObservationForGym, NormalizeReward as NormalizeRewardForGym

from gymnasium.wrappers import NormalizeObservation, NormalizeReward # TODO should i nml the Reward?
import grid2op
from grid2op.Environment import MultiEnvMultiProcess
from grid2op.gym_compat import GymEnv, BoxGymObsSpace, BoxGymActSpace, DiscreteActSpace # if we import gymnasium, GymEnv will convert to Gymnasium!   
from grid2op.Reward import (
    CombinedReward, 
    IncreasingFlatReward, 
    DistanceReward, 
    RedispReward, 
    N1Reward, 
    L2RPNReward,
    FlatReward
)
from lightsim2grid import LightSimBackend

from hungto.ExternalLibs.RL2Grid_v0.RL2Grid.common.imports import np, gym
from hungto.ExternalLibs.RL2Grid_v0.RL2Grid.common.reward import (
    LineMarginReward,
    RedispRewardv1,
    L2RPNRewardRegularized,
    LineRootMarginReward,
    LineCubeRootReward,
    LineCubeRootRewardNonRegularized,
)


from .heuristic import (
    GridOpRecoAndRevertBus, 
    GridOpIdle, 
    GridOpIdleNonLoop, 
    GridOpIdleForGym
)

ENV_DIR = os.path.dirname(__file__)



REWARD_CLASS_MAP = {
    "IncreasingFlatReward": IncreasingFlatReward,
    "TopologyReward": DistanceReward,
    "LineMarginReward": LineMarginReward,
    "RedispRewardv1": RedispRewardv1,
    "L2RPNRewardRegularized": L2RPNRewardRegularized,
    "LineRootMarginReward": LineRootMarginReward,

    "LineCubeRootReward": LineCubeRootReward,
    "LineCubeRootRewardNonRegularized": LineCubeRootRewardNonRegularized,
  
    "L2RPNReward": L2RPNReward,
    "N1Reward": N1Reward, # Note: N1Reward needs l_id, not easily configurable via simple CLI string yet.
    "FlatReward": FlatReward,
  
    }



def load_config(file_path):

    # Get the directory of the current module (__file__ contains the path of the current file)
    with open(f"{ENV_DIR}/{file_path}", 'r') as file:
        config = json.load(file)
    return config

def norm_action_limits(gym_env, attrs):
    # Getting the right coefficients to have action limits in [0, 1] and use sigmoid output activations4sdrub4lE
    # (following how grid2op handles mult and add factors: https://github.com/rte-france/Grid2Op/blob/5d938584da3c42dc26fc8128335e20ef382bef00/grid2op/gym_compat/box_gym_actspace.py#L376)
    mult_factor, add_factor = {}, {}
    for attr in attrs:
        attr_box = gym_env.action_space[attr]

        # Filter out elements where both low and high limits are not equal to 0
        feasible_acts = (attr_box.low != attr_box.high)
        low = attr_box.low[feasible_acts]
        high = attr_box.high[feasible_acts]
        
        # Calculate range, multiplicative factor, and additive factor
        range = high - low
        mult_factor[attr] = range
        add_factor[attr] = low

    return mult_factor, add_factor


def _configure_rewards(cr, g2op_env, args, default_reward_setup_func):
    """
    Configures rewards for the environment.
    Prioritizes CLI arguments if provided, otherwise uses the default_reward_setup_func.
    """
    env_type = args.action_type.lower()

    if args.reward_fn and args.reward_factors:
        if len(args.reward_fn) != len(args.reward_factors):
            raise ValueError(
                f"Number of reward functions ({len(args.reward_fn)}: {args.reward_fn}) must match "
                f"number of reward factors ({len(args.reward_factors)}: {args.reward_factors})."
            )

        for fn_name, factor in zip(args.reward_fn, args.reward_factors):
            if fn_name in REWARD_CLASS_MAP:
                RewardClass = REWARD_CLASS_MAP[fn_name]
                reward_instance = None
                try:
                    if fn_name == "IncreasingFlatReward":
                        reward_instance = RewardClass(per_timestep=1/g2op_env.chronics_handler.max_episode_duration())
                    elif fn_name == "FlatReward":
                        reward_instance = RewardClass(per_timestep=1)
                    elif fn_name in ["LineCubeRootReward", "LineCubeRootRewardNonRegularized"]:
                        reward_instance = RewardClass(
                            n_th_root_safe=args.reward_param_lsmrm_n_safe,
                            n_th_root_overflow=args.reward_param_lsmrm_n_overflow
                        )


                   
                    else: # For rewards with no specific constructor args from our list
                        reward_instance = RewardClass()

                    if reward_instance:
                        cr.addReward(fn_name, reward_instance, float(factor))
                except Exception as e:
                    print(f"Warning: Could not instantiate or add reward function '{fn_name}' with factor {factor}. Error: {e}. Skipping.")
            else:
                print(f"Warning: Reward function '{fn_name}' not found in REWARD_CLASS_MAP. Skipping.")
    else:
        default_reward_setup_func(cr, g2op_env, env_type, args) # Pass args for reward-specific defaults

    cr.initialize(g2op_env)

def _default_rewards_make_env(cr, g2op_env, env_type, args):
    cr.addReward("IncreasingFlatReward",
                IncreasingFlatReward(per_timestep=1/g2op_env.chronics_handler.max_episode_duration()),
                0.1)
    if env_type == 'topology':
       cr.addReward("TopologyReward", DistanceReward(), 0.3)
    cr.addReward("redispatchReward", RedispRewardv1(), 0.3 if env_type == 'topology' else 0.6)
    cr.addReward("LineMarginReward", LineMarginReward(), 0.3)


def make_env(args, idx, resume_run=False, generate_class=False, async_vec_env=False, params=None, eval_mode=False, deactivate_forecast=True):
    def thunk():

        config = load_config(args.env_config_path)
        env_id =  args.eval_env_id if eval_mode else args.env_id   #args.eval_env_id if eval_mode else args.env_id     bus14_test
        print(f"=== Creating environment: {env_id} ===")
        env_type = args.action_type.lower()
        difficulty = args.difficulty

        env_config = config['environments']
        assert env_id in env_config.keys(), f"Invalid environment ID: {env_id}. Available IDs are: {env_config.keys()}"

        env_types = ["topology", "redispatch"]
        assert env_type in env_types, f"Invalid environment type: {env_type}. Available IDs are: {env_types}"

        max_difficulty = env_config[env_id]['difficulty']
        assert difficulty < max_difficulty, f"Invalid difficulty: {difficulty}. Difficulty limit is : {max_difficulty-1}"
           
        g2op_env = grid2op.make(
            env_config[env_id]['grid2op_id'], 
            reward_class=CombinedReward, 
            #experimental_read_from_local_dir=True if async_vec_env else False,
            #other_rewards={f"line_{l_id}": N1Reward(l_id=l_id) for l_id in range(186)}
            backend=LightSimBackend(),      # detailed_infos_for_cascading_failures=True
            #class_in_file=True               #only for grid2op>=1.10.3
            experimental_read_from_local_dir=True,     #for grid2op<1.10.3  MUST do this to handle pickling? but why
        ) 
        
        cr = g2op_env.get_reward_instance()
        _configure_rewards(cr, g2op_env, args, _default_rewards_make_env)

        if deactivate_forecast:
            g2op_env.deactivate_forecast() # will run faster 15%

        g2op_env.chronics_handler.set_chunk_size(100)    # Instead of loading all episode data, get chunks of 100
        if generate_class:
            g2op_env.generate_classes()
            print("Class generated offline for AsyncVecEnv execution")
            quit()
        
        gym_env = GymEnv(g2op_env, shuffle_chronics=True)      # NOTE: access the g2op_env via gym_env.init_env

        # Apply parameters to the environment
        if params is not None:
            # Use the Parameters object passed from main.py
            gym_env.init_env.change_parameters(params)
        else:
            # Making sure we can act on 1 sub / line status at the same step (default behavior if no params passed)
            p = gym_env.init_env.parameters
            p.MAX_LINE_STATUS_CHANGED = 1 
            p.MAX_SUB_CHANGED = 1 
            gym_env.init_env.change_parameters(p)

        # The .reset is required to change the parameters
        if resume_run: gym_env.reset()   # NOTE: to use set_id on gym_env.init_env, we first have to set gym_env's seed with a reset   
        else:
            gym_env.reset(seed=args.seed+idx)

        gym_env.init_env.chronics_handler.shuffle()
        #print(f"id: {gym_env.init_env.chronics_handler.get_id()}")

        # Prepare action and observation spaces
        state_attrs = config['state_attrs']
        obs_attrs = state_attrs['default']
        if env_config[env_id]['maintenance']: obs_attrs += state_attrs['maintenance']
        #if env_config[env_id]['opponent']: obs_attrs += state_attrs['curtailment']

        if env_type == 'topology': 
            obs_attrs += state_attrs['topology']

            # Set the actions space from the loaded list of (vectorized) actions
            loaded_action_space = np.load(f"{ENV_DIR}/action_spaces/{env_id}_action_space.npy", allow_pickle=True)
            #print(f"Loaded action space: {loaded_action_space}")
            
            # Increase the action_space size exponentially (from 50) based on difficulty
            n_actions = np.geomspace(50, len(loaded_action_space), num=max_difficulty).astype(int)
            gym_env.action_space = DiscreteActSpace(
                g2op_env.action_space,
                action_list=loaded_action_space[:n_actions[difficulty]]
            )

            ### emarche ###
            '''
            # Get the number of elements connected to each substation in a friend copy-paste format
            print(gym_env.init_env.n_sub)
            for i in gym_env.init_env.sub_info: 
                print(i)
            quit()
            
            # Get the n° of actions for each substation, based on the difficulty level
            for d in range(max_difficulty):
                gym_env.action_space = DiscreteActSpace(
                    g2op_env.action_space,
                    action_list=loaded_action_space[:n_actions[d]]
                )
                print(f"Difficulty set to {d}, n° actions: {gym_env.action_space.n}")
                sub_actions = np.zeros(gym_env.init_env.n_sub)
                line_actions = np.zeros(gym_env.init_env.n_line)
                for action in range(1, gym_env.action_space.n):     # ignoring 0 because it's do_nothing
                    dict_action = gym_env.action_space.from_gym(action).as_dict()
                    if 'change_bus_vect' in dict_action:
                        sub_actions[int(dict_action['change_bus_vect']['modif_subs_id'][0])] += 1
                    elif 'change_line_status' in dict_action:
                        line_actions[int(dict_action['change_line_status']['changed_id'][0])] += 1
                       
                print("Substations")
                for i in sub_actions:
                    print(int(i))
                print(f"Total substation actions = {np.sum(sub_actions, dtype=int)}/{gym_env.action_space.n} = {np.sum(sub_actions, dtype=int)/gym_env.action_space.n}")

                #print("Lines")
                #for i in line_actions:
                #    print(i)
                print(f"Total line actions = {np.sum(line_actions, dtype=int)}/{gym_env.action_space.n} = {np.sum(line_actions, dtype=int)/gym_env.action_space.n}")
                print()
            quit()
            '''
            ##############
        else:
            actions_to_keep = ['redispatch', 'curtail']

            obs_attrs += state_attrs['redispatch']
            if env_config[env_id]['renewable']: 
                obs_attrs += state_attrs['curtailment']
            if env_config[env_id]['battery']:
                obs_attrs += state_attrs['storage']
                actions_to_keep += ['set_storage']
        
            mult_factor, add_factor = norm_action_limits(gym_env, actions_to_keep)

            gym_env.action_space = BoxGymActSpace(
                gym_env.init_env.action_space,
                attr_to_keep=actions_to_keep,
                multiply=mult_factor,   
                add=add_factor
            )
        # Set the observation space
        gym_env.observation_space = BoxGymObsSpace(gym_env.init_env.observation_space,
                                            attr_to_keep=obs_attrs,
                                            #divide={"gen_p": gym_env.init_env.gen_pmax,
                                            #        "actual_dispatch": gym_env.init_env.gen_pmax},
        )


        if args.norm_obs: 
            gym_env = NormalizeObservation(gym_env)      

        # Add reward normalization if requested
        if hasattr(args, "norm_reward") and args.norm_reward:
            gym_env = NormalizeReward(gym_env)

        if args.use_heuristic:
            if args.heuristic_type == "idle":
                gym_env = GridOpIdle(gym_env)
            elif args.heuristic_type == "reco_revert":
                gym_env = GridOpRecoAndRevertBus(gym_env)
            elif args.heuristic_type == "idle_non_loop":
                gym_env = GridOpIdleNonLoop(gym_env)
        else:
            gym_env = gym.wrappers.RecordEpisodeStatistics(gym_env)






        return gym_env 

    # Return the environment object with custom serialization methods
    return thunk













































