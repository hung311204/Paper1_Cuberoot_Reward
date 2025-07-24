# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from .imports import np

from grid2op.Reward.baseReward import BaseReward
from grid2op.Reward import RedispReward
from grid2op.Reward import L2RPNReward
from grid2op.dtypes import dt_float

class LineMarginReward(BaseReward):
    def __init__(self, logger=None):
        super().__init__(logger=logger)
        self.penalty = dt_float(-1.0)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error:
            return self.penalty * sum(~env.current_obs.line_status) / env.current_obs.n_line
        
        if is_illegal or is_ambiguous:
            return 0.0

        ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
        thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
        margin = np.divide(thermal_limits - ampere_flows, thermal_limits + 1e-10)

        # Reward is based on how much lines are used (the lower the better and goes negative in case of overflow) and is penalized for disconnected lines. We then normalize everything between (more or less) [-1, 1]  
        reward = margin[env.current_obs.line_status].sum() + (self.penalty * sum(~env.current_obs.line_status))

        return reward / env.current_obs.n_line
      
class RedispRewardv1(RedispReward):
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if is_done:
            # if the episode is over and it's my fault (i did a blackout) i strongly
            if has_error or is_illegal or is_ambiguous:
                return self.reward_min
        elif is_illegal or is_ambiguous:
            return self._reward_illegal_ambiguous

        # compute the losses
        gen_p, *_ = env.backend.generators_info()
        load_p, *_ = env.backend.loads_info()
        # don't forget to convert MW to MWh !
        losses = (gen_p.sum() - load_p.sum()) * env.delta_time_seconds / 3600.0
        # compute the marginal cost
        gen_activeprod_t = env._gen_activeprod_t
        marginal_cost = np.max(env.gen_cost_per_MW[gen_activeprod_t > 0.0])
        # redispatching amount
        actual_dispatch = env._actual_dispatch
        redisp_cost = (
            self._alpha_redisp * np.abs(actual_dispatch).sum() * marginal_cost * env.delta_time_seconds / 3600.0
        )
        
        # cost of losses
        losses_cost = losses * marginal_cost

        # cost of storage
        c_storage = np.abs(env._storage_power).sum() * marginal_cost * env.delta_time_seconds / 3600.0
        
        # total "regret"
        regret = losses_cost + redisp_cost + c_storage

        # compute reward and normalize
        reward = dt_float(-regret/self.max_regret)

        return reward

class L2RPNRewardRegularized(L2RPNReward):
    """
    This reward class extends the original L2RPNReward by normalizing the result
    by dividing it by the number of lines in the grid.
    
    This normalization ensures that the reward scale remains consistent regardless
    of the grid size, making it more suitable for comparing performance across
    different grid topologies or for curriculum learning scenarios.
    
    The original L2RPNReward computes the sum of squared margins for each power line,
    where the margin is defined as (thermal limit - flow) / thermal limit when flow
    is below the thermal limit, and 0 otherwise.
    
    This version normalizes that reward by dividing by the number of lines, so the
    range becomes [0, 1] instead of [0, n_line].
    """
    
    def __init__(self, logger=None):
        super().__init__(logger=logger)
    
    def initialize(self, env):
        # Initialize parent class
        super().initialize(env)
        # Override min/max rewards for normalization
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)  # Instead of n_line in the parent class
        self.n_line = env.backend.n_line
    
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        # Get the original reward calculation from parent class
        reward = super().__call__(action, env, has_error, is_done, is_illegal, is_ambiguous)
        
        # Normalize the reward by dividing by the number of lines
        if not is_done and not has_error:
            reward = reward / self.n_line
        
        return reward



class LineRootMarginRewardLegacyLegacy(BaseReward):
    """
    This reward also calculates and punishes based on margin of each line like L2RPNReward,
    or LineMarginReward of RL2Grid, but based on the root of the margin. 
    
    The margin is defined, for each powerline as:
    `margin of a powerline = (thermal limit - flow in amps) / thermal limit`
    if 0 < flow in amps <= HARD_OVERFLOW_THRESHOLD * thermal limit) else `margin of a powerline  = -1`
    (0 < flow in amps means that if the line is disconnected, the margin is -1)
    
    Note that in the L2RPNReward, the condition is: 
    if flow in amps <= thermal limit) else `margin of a powerline  = 0.`,
    but we don't do that because we want to consider rho in the range (0, 2] 
    (Hard OT is always 2.0 in every Grid2Op environments), not just [0, 1] like in the L2RPNReward, 
    so that we can punish the lines that has rho > 1 with a negative reward.

    Root_point = (margin)**(1/n), 
    n depends on your choice. 
    We use n = 5 in this implementation.
    
    Then the reward is the sum of the root point of all lines, 
    and is normalized by the number of lines:
    Reward = sum(Root point of this line) / number of lines, for each powerline
    
    This way, the reward interval is [-1, 1] and it will be super steep when rho is around 1 (Root point's derivative = negative infinity when rho = 1), which is what we want.

    """
    # now that i've describe it, pls implement it, follow the way of L2RPNReward if you can

    def __init__(self, logger=None, n_th_root: int = 5):
        BaseReward.__init__(self, logger=logger)
        self.n_th_root = n_th_root
        self.root_power = 1.0 / self.n_th_root

    def initialize(self, env):
        self.reward_min = dt_float(-1.0)
        self.reward_max = dt_float(1.0)  
        self.penalty = dt_float(-1.0)
        self.n_line = env.backend.n_line
  
        # TODO what if Hard OT is not 2.0?
        if hasattr(env, 'parameters') and hasattr(env.parameters, 'HARD_OVERFLOW_THRESHOLD'):
            self.hard_overflow_threshold_factor = dt_float(env.parameters.HARD_OVERFLOW_THRESHOLD)
            print(f"HARD_OVERFLOW_THRESHOLD found in env.parameters: {self.hard_overflow_threshold_factor}")
        else:
            self.logger.warning(
                "HARD_OVERFLOW_THRESHOLD not found in env.parameters. Defaulting to 2.0. "
                "Ensure this aligns with your environment's settings."
            )
            self.hard_overflow_threshold_factor = dt_float(2.0)


    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if not is_done and not has_error:
            # Get line flows and thermal limits
            ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
            thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
            
            # Calculate rho (flow/limit ratio) for each line
            rho = np.divide(ampere_flows, thermal_limits + 1e-8)
            
            # Initialize root_point array
            root_point = np.zeros_like(rho, dtype=dt_float)
            
            # For lines with rho <= 1.0, use n-th root of margin
            mask_safe = rho <= 1.0
            margin_safe = 1.0 - rho[mask_safe]
            root_point[mask_safe] = np.power(margin_safe, self.root_power)
            
            # For lines with rho > 1.0, use linear margin divided by root point at hard OT
            mask_overflow = ~mask_safe
            margin_overflow = 1.0 - rho[mask_overflow]
            # Calculate root point at hard OT for normalization
            hard_ot_margin = 1.0 - self.hard_overflow_threshold_factor
            hard_ot_root_point = np.abs(hard_ot_margin)  # Make sure it's positive
            #hard_ot_root_point = np.abs(np.power(hard_ot_root_point, self.root_power))
            # Linear, why not?
            root_point[mask_overflow] = margin_overflow / hard_ot_root_point
            #root_point[mask_overflow] = np.power(margin_overflow, self.root_power) / hard_ot_margin
            
            # Sum rewards for connected lines and add penalties for disconnected lines
            connected_lines_reward = root_point[env.current_obs.line_status].sum()
            disconnected_lines_penalty = self.penalty * sum(~env.current_obs.line_status)
            
            # Normalize by total number of lines
            res = (connected_lines_reward + disconnected_lines_penalty) / self.n_line
        else:
            res = self.reward_min

        return res

class LineRootMarginReward(BaseReward):
    """
    This reward also calculates and punishes based on margin of each line like L2RPNReward,
    or LineMarginReward of RL2Grid, but based on the root of the margin. 
    
    The margin is defined, for each powerline as:
    `margin of a powerline = (thermal limit - flow in amps) / thermal limit`
    if 0 < flow in amps <= HARD_OVERFLOW_THRESHOLD * thermal limit) else `margin of a powerline  = -1`
    (0 < flow in amps means that if the line is disconnected, the margin is -1)
    
    Note that in the L2RPNReward, the condition is: 
    if flow in amps <= thermal limit) else `margin of a powerline  = 0.`,
    but we don't do that because we want to consider rho in the range (0, 2] 
    (Hard OT is always 2.0 in every Grid2Op environments), not just [0, 1] like in the L2RPNReward, 
    so that we can punish the lines that has rho > 1 with a negative reward.

    Root_point = (margin)**(1/n), 
    n depends on your choice. 
    We use n = 5 in this implementation.
    
    Then the reward is the sum of the root point of all lines, 
    and is normalized by the number of lines:
    Reward = sum(Root point of this line) / number of lines, for each powerline
    
    This way, the reward interval is [-1, 1] and it will be super steep when rho is around 1 (Root point's derivative = negative infinity when rho = 1), which is what we want.

    """
    # now that i've describe it, pls implement it, follow the way of L2RPNReward if you can

    def __init__(self, logger=None, n_th_root: int = 5):
        BaseReward.__init__(self, logger=logger)
        self.n_th_root = n_th_root
        self.root_power = 1.0 / self.n_th_root

    def initialize(self, env):
        self.reward_min = dt_float(-1.0)
        self.reward_max = dt_float(1.0)  
        self.penalty = dt_float(-1.0)
        self.n_line = env.backend.n_line
  
        # TODO what if Hard OT is not 2.0?
        if hasattr(env, 'parameters') and hasattr(env.parameters, 'HARD_OVERFLOW_THRESHOLD'):
            self.hard_overflow_threshold_factor = dt_float(env.parameters.HARD_OVERFLOW_THRESHOLD)
            print(f"HARD_OVERFLOW_THRESHOLD found in env.parameters: {self.hard_overflow_threshold_factor}")
        else:
            self.logger.warning(
                "HARD_OVERFLOW_THRESHOLD not found in env.parameters. Defaulting to 2.0. "
                "Ensure this aligns with your environment's settings."
            )
            self.hard_overflow_threshold_factor = dt_float(2.0)


    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if not is_done and not has_error:
            # Get line flows and thermal limits
            ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
            thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
            
            # Calculate rho (flow/limit ratio) for each line
            rho = np.divide(ampere_flows, thermal_limits + 1e-8)
            
            # Initialize root_point array
            root_point = np.zeros_like(rho, dtype=dt_float)
            
            # For lines with rho <= 1.0, use n-th root of margin
            mask_safe = rho <= 1.0
            margin_safe = 1.0 - rho[mask_safe]
            root_point[mask_safe] = np.power(margin_safe, self.root_power)
            
            # For lines with rho > 1.0, use linear margin divided by root point at hard OT
            mask_overflow = ~mask_safe
            margin_overflow = 1.0 - rho[mask_overflow]
            # Calculate root point at hard OT for normalization
            hard_ot_margin = 1.0 - self.hard_overflow_threshold_factor
            hard_ot_root_point = np.abs(hard_ot_margin)  # Make sure it's positive
            #hard_ot_root_point = np.abs(np.power(hard_ot_root_point, self.root_power))
            # Linear, why not?
            root_point[mask_overflow] = margin_overflow / hard_ot_root_point
            #root_point[mask_overflow] = np.power(margin_overflow, self.root_power) / hard_ot_margin
            
            # Sum rewards for connected lines and add penalties for disconnected lines
            connected_lines_reward = root_point[env.current_obs.line_status].sum()
            disconnected_lines_penalty = self.penalty * sum(~env.current_obs.line_status)
            
            # Normalize by total number of lines
            res = (connected_lines_reward + disconnected_lines_penalty) / self.n_line
        else:
            res = self.reward_min

        return res
    



class LineCubeRootReward(BaseReward):
    """
       | Hung |
    """
    def __init__(self, logger=None, 
                 n_th_root_safe: int = 5,
                 n_th_root_overflow: int = 5):
        BaseReward.__init__(self, logger=logger)
        self.n_th_root_safe = n_th_root_safe
        self.n_th_root_overflow = n_th_root_overflow
        self.root_power_safe = 1.0 / self.n_th_root_safe
        self.root_power_overflow = 1.0 / self.n_th_root_overflow

    def initialize(self, env):
        self.reward_min = dt_float(-1.0)
        self.reward_max = dt_float(1.0)  
        self.penalty = dt_float(-1.0)
        self.n_line = env.backend.n_line
  
        # TO_DO what if Hard OT is not 2.0?
        if hasattr(env, 'parameters') and hasattr(env.parameters, 'HARD_OVERFLOW_THRESHOLD'):
            self.hard_overflow_threshold_factor = dt_float(env.parameters.HARD_OVERFLOW_THRESHOLD)
            print(f"HARD_OVERFLOW_THRESHOLD found in env.parameters: {self.hard_overflow_threshold_factor}")
        else:
            self.logger.warning(
                "HARD_OVERFLOW_THRESHOLD not found in env.parameters. Defaulting to 2.0. "
                "Ensure this aligns with your environment's settings."
            )
            self.hard_overflow_threshold_factor = dt_float(2.0)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if not is_done and not has_error:
            # Get line flows and thermal limits
            ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
            thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
            
            # Calculate rho (flow/limit ratio) for each line
            rho = np.divide(ampere_flows, thermal_limits + 1e-8)
            
            # Initialize root_point array
            root_point = np.zeros_like(rho, dtype=dt_float)
            
            # For lines with rho <= 1.0, use n-th root of margin
            mask_safe = rho <= 1.0
            margin_safe = 1.0 - rho[mask_safe]
            root_point[mask_safe] = np.power(margin_safe, self.root_power_safe)
            
            # For lines with rho > 1.0, use n-th root of margin and normalize using Hard OT
            mask_overflow = ~mask_safe
            margin_overflow = 1.0 - rho[mask_overflow]
            
            # Calculate root point at hard OT for normalization
            hard_ot_margin = 1.0 - self.hard_overflow_threshold_factor
            hard_ot_root_point = np.power(np.abs(hard_ot_margin), self.root_power_overflow)  # make sure it is positive
            # Normalize overflow margins using hard OT root point
            root_point[mask_overflow] = np.sign(margin_overflow) * np.power(np.abs(margin_overflow), self.root_power_overflow) / hard_ot_root_point
            
            # Set penalty for disconnected lines
            root_point[~env.current_obs.line_status] = self.penalty
            
            # Average over all lines
            res = root_point.sum() / self.n_line
        else:
            res = self.reward_min

        return res





class LineCubeRootRewardNonRegularized(LineCubeRootReward):
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        base_reward = super().__call__(action, env, has_error, is_done, is_illegal, is_ambiguous)
        return base_reward * self.n_line