from abc import ABC, abstractmethod
from hungto.ExternalLibs.RL2Grid_v0.RL2Grid.common.imports import gym, np
# Get the substations impacted by an action (as a True/False mask with n_sub dimension)
#   env.action_space.from_gym(3)._subs_impacted
from collections import deque
# Get the idx of an object in the topo_vect
#   obs.line_or_pos_topo_vect
#   obs.line_ex_pos_topo_vect
#   obs.load_pos_topo_vect
#   obs.gen_pos_topo_vect
#   obs.storage_pos_topo_vect

# Get the bus on which the objects are connected
#   obs.topo_vect[obs.line_or_pos_topo_vect]
RHO_SAFETY_THRESHOLD = 0.95

class GridOp(gym.Wrapper, ABC):
    @property
    def _risk_overflow(self):
        return self.init_env.current_obs.rho.max() >= RHO_SAFETY_THRESHOLD

    @property  
    def _obs(self):
        return self.init_env.current_obs
    
    @abstractmethod
    def _get_heuristic_actions(self):
        return []

    def apply_actions(self):
        use_heuristic, heuristic_reward = True, 0.
        
        while use_heuristic:
            g2o_actions = self._get_heuristic_actions()
            if not g2o_actions: break
            for g2o_act in g2o_actions:
                _, reward, done, info = self.init_env.step(g2o_act)
                heuristic_reward += reward    # Cumulate reward over heuristic steps
                #print(f"HEURISTIC: {g2o_actions[0]} -> {reward} -> {heuristic_reward}")  ##PP
                
                 
                if done or self._risk_overflow:   # Resume the agent if in a risky situation
                    use_heuristic = False
                    break

        return heuristic_reward, done, info

    def step(self, gym_action):
        #print(self.action_space.from_gym(gym_action)) ##PP 
        _, reward, done, info = self.init_env.step(self.action_space.from_gym(gym_action))
        if not done and not self._risk_overflow:
            heuristic_reward, done, info = self.apply_actions()
            reward += heuristic_reward
        self.ep_reward += reward
        gym_obs = self.observation_space.to_gym(self._obs)

        if done:
            info['episode'] = {'l': [self.init_env.nb_time_step], 'r': [self.ep_reward]}    # replacing the use of RecordEpisodeStatistics
            #print(info)    ##PP
            #print(info[detailed_infos_for_cascading_failures])    ##PP
        return gym_obs, float(reward), done, False, info    # Truncation is always false in g2o envs
        
    def reset(self, **kwargs):
        super().reset(**kwargs)  # Reset the underlying scenario
        self.ep_reward = 0.

        info = {}
        if not self._risk_overflow:
            _, _, info = self.apply_actions()  # First reward is zero
        
        return self.observation_space.to_gym(self._obs), info

class GridOpIdle(GridOp):
    def _get_heuristic_actions(self):      
        if self._risk_overflow: return []
        else: return [self.init_env.action_space()]    
        
class GridOpReco(GridOp):
    def _get_line_reconnect_actions(self):
        to_reconnect = (~self._obs.line_status) & (self._obs.time_before_cooldown_line == 0)
        if np.any(to_reconnect):
            reco_id = np.where(to_reconnect)[0]     # Reconnect lines that are out of cooldown
            return [self.init_env.action_space({"set_line_status": [(line_id, 1)]}) for line_id in reco_id]
        return []
    
    def _get_heuristic_actions(self):
        if self._risk_overflow: return []
        actions = self._get_line_reconnect_actions()
        if np.any(actions): return actions
        return [self.init_env.action_space()]    

class GridOpRevertBus(GridOp):
    def _get_bus_revert_actions(self):
        # For each substation, count the objects that have changed bus 
        bus_changed_info = np.array([
            np.count_nonzero(sub_buses > 1) 
             for sub_buses in np.split(self._obs.topo_vect, np.cumsum(self._obs.sub_info[:-1]))
        ])

        # Get the idx of most changed substation that is out of cooldown
        # Get the subs out of cooldown where there have been some bus changes
        revertable_subs = bus_changed_info[(self._obs.time_before_cooldown_sub == 0) & (bus_changed_info > 0)]      

        if revertable_subs.size > 0:    # if there is a sub to revert           
            # Get unique values in revertable_subs and their counts
            unique_subs, counts = np.unique(revertable_subs, return_counts=True)

            # Create a dictionary to map those unique values to their inverted counts (descending) - handles duplicates
            ordered_sub_values = dict(zip(unique_subs, np.argsort(counts)[::-1]))

            # Find indexes where elements in bus_changed_info match values in revertable_subs
            bus_changed_idxs = np.where(np.in1d(bus_changed_info, unique_subs))[0]

            # Get the corresponding order for each matching element using the dictionary
            bus_changed_orders = [ordered_sub_values[sub] for sub in bus_changed_info[bus_changed_idxs]]
            # Sort and rank indexes based on the orders (descending order)
            bus_changed_idxs = bus_changed_idxs[np.argsort(bus_changed_orders)]
            
            return [self.init_env.action_space(
                {"set_bus": 
                 {"substations_id": [(revert_sub_idx, np.ones(self._obs.sub_info[revert_sub_idx], dtype=int))]}})
                for revert_sub_idx in bus_changed_idxs
            ]
        return []
    
    def _get_heuristic_actions(self):    
        if self._risk_overflow: return []
        actions = self._get_bus_revert_actions()
        if np.any(actions): return actions
        return [self.init_env.action_space()]    

class GridOpRecoAndRevertBus(GridOpReco, GridOpRevertBus):        
    def _get_heuristic_actions(self):   
        if self._risk_overflow: return []

        actions = self._get_line_reconnect_actions()
        actions.extend(self._get_bus_revert_actions())
      
        if actions: return actions
        return [self.init_env.action_space()]    
    



    # ============ NON LOOP =======================================================


    
# NEW CLASS: GridOpNonLoop
class GridOpNonLoop(GridOp):
    """
    A GridOp wrapper that applies at most one heuristic action per call to `apply_actions`.
    It does not loop to take as many heuristic actions as possible.
    """
    def apply_actions(self):
        """
        Tries to apply at most one heuristic action.
        Returns the reward from the heuristic action, and the resulting done state and info.
        If no heuristic action is taken, reward is 0, done is False (as it's called when not done),
        and info is an empty dictionary.
        """
        use_heuristic, heuristic_reward = True, 0.

        g2o_actions = self._get_heuristic_actions() # Get proposed heuristic actions
        
        if g2o_actions:  # If there are any heuristic actions proposed
            g2o_act = g2o_actions[0]  # Take only the first one from the list
            
            # Execute the single heuristic step
            _, reward, done, info = self.init_env.step(g2o_act)
            heuristic_reward += reward    # Cumulate reward over heuristic steps
            #print(f"HEURISTIC: {g2o_actions[0]} -> {reward} -> {heuristic_reward}")  ##PP
            
                
        return heuristic_reward, done, info


    def step(self, gym_action):
        # Check risk BEFORE applying agent action
        if not self._risk_overflow:
            # If risky, ignore agent action and take a do-nothing step
            # The environment will handle the risk internally or fail
            # print("RISKY STATE DETECTED BEFORE AGENT ACTION. TAKING DO-NOTHING STEP.") # Optional debug
            _, reward, done, info = self.init_env.step(self.init_env.action_space())
            # No heuristic actions are applied after this step because the state was risky
        else:
            # If not risky, apply the agent's action
            # print("SAFE STATE. APPLYING AGENT ACTION.") # Optional debug
            #print(self.action_space.from_gym(gym_action)) ##PP
            _, reward, done, info = self.init_env.step(self.action_space.from_gym(gym_action))

        self.ep_reward += reward
        gym_obs = self.observation_space.to_gym(self._obs)

        if done:
            info['episode'] = {'l': [self.init_env.nb_time_step], 'r': [self.ep_reward]}    # replacing the use of RecordEpisodeStatistics
            #print(info)    ##PP
            #print(info[detailed_infos_for_cascading_failures])    ##PP
        return gym_obs, float(reward), done, False, info    # Truncation is always false in g2o envs
        


class GridOpIdleNonLoop(GridOpNonLoop):
    def _get_heuristic_actions(self):      
        if self._risk_overflow: return []
        else: return [self.init_env.action_space()]    


class GridOpForGym(gym.Wrapper, ABC):
    @property
    def _risk_overflow(self):
        return self.init_env.current_obs.rho.max() >= RHO_SAFETY_THRESHOLD

    @property  
    def _obs(self):
        return self.init_env.current_obs
    
    @abstractmethod
    def _get_heuristic_actions(self):
        return []

    def apply_actions(self):
        use_heuristic, heuristic_reward = True, 0.
        
        while use_heuristic:
            g2o_actions = self._get_heuristic_actions()
            if not g2o_actions: break
            for g2o_act in g2o_actions:
                _, reward, done, info = self.init_env.step(g2o_act)
                heuristic_reward += reward    # Cumulate reward over heuristic steps
                #print(f"HEURISTIC: {g2o_actions[0]} -> {reward} -> {heuristic_reward}")  ##PP
                
                 
                if done or self._risk_overflow:   # Resume the agent if in a risky situation
                    use_heuristic = False
                    break

        return heuristic_reward, done, info

    def step(self, gym_action):
        #print(self.action_space.from_gym(gym_action)) ##PP 
        _, reward, done, info = self.init_env.step(self.action_space.from_gym(gym_action))
        if not done and not self._risk_overflow:
            heuristic_reward, done, info = self.apply_actions()
            reward += heuristic_reward
        
        # Do this if we want to clipp reward for meta RL 
        # and also affect the tracked "Return" by RecordEpisodeStatistics, 
        # for now "Return" is not affected by the clipping
        #reward = min(reward, 1.0)
        
        # Or maybe i can even create a new attribute to store 
        # the clipped Return?, for example self.ep_reward_clipped? like this:
        #clipped_reward = min(reward, 1.0)
        #self.ep_reward_clipped += clipped_reward

        self.ep_reward += reward
        gym_obs = self.observation_space.to_gym(self._obs)

        if done:
            info['episode'] = {'l': [self.init_env.nb_time_step], 'r': [self.ep_reward]}    # replacing the use of RecordEpisodeStatistics
            #print(info)    ##PP
            #print(info[detailed_infos_for_cascading_failures])    ##PP
        return gym_obs, float(reward), done, False, info    # Truncation is always false in g2o envs
        
    def reset(self, **kwargs):
        super().reset(**kwargs)  # Reset the underlying scenario
        self.ep_reward = 0.

        info = {}
        if not self._risk_overflow:
            _, _, info = self.apply_actions()  # First reward is zero
        
        return self.observation_space.to_gym(self._obs), info # for gym, only obs, no info 

class GridOpIdleForGym(GridOpForGym):
    def _get_heuristic_actions(self):      
        if self._risk_overflow: return []
        else: return [self.init_env.action_space()]   





   