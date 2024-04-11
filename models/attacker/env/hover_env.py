"""QuadX Hover Environment for Attacker model."""
import numpy as np
from math import log10, sqrt
from env.base_env import QuadXBaseEnv as QuadXBase


class QuadXHoverAgiliciousAttEnv(QuadXBase):
    """Simple Hover Environment for Attacker model.

    Actions are vp, vq, vr, T, ie: angular rates and thrust.
    The target is to not crash for the longest time possible.

    Args:
        sparse_reward (bool): whether to use sparse rewards or not.
        flight_dome_size (float): size of the allowable flying area.
        max_duration_seconds (float): maximum simulation time of the environment.
        angle_representation (str): can be "euler" or "quaternion".
        agent_hz (int): looprate of the agent to environment interaction.
        render_mode (None | str): can be "human" or None.
        render_resolution (tuple[int, int]): render_resolution.
    """

    def __init__(
        self,
        sparse_reward: bool = False,
        fixed_hover_position = None,
        flight_dome_size: float = 3.0,
        max_duration_seconds: float = 10.0,
        angle_representation: str = "quaternion",
        agent_hz: int = 120,
        render_mode: None | str = None,
        render_resolution: tuple[int, int] = (480, 480),
    ):
        """__init__.

        Args:
            sparse_reward (bool): whether to use sparse rewards or not.
            flight_dome_size (float): size of the allowable flying area.
            max_duration_seconds (float): maximum simulation time of the environment.
            angle_representation (str): can be "euler" or "quaternion".
            agent_hz (int): looprate of the agent to environment interaction.
            render_mode (None | str): can be "human" or None.
            render_resolution (tuple[int, int]): render_resolution.
        """
        super().__init__(
            flight_dome_size=flight_dome_size,
            start_pos=np.array([[0.0, 0.0, 0.5]]),
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

        """GYMNASIUM STUFF"""
        self.observation_space = self.combined_space

        """ ENVIRONMENT CONSTANTS """
        self.sparse_reward  = sparse_reward
        self.fixed_hover_position = fixed_hover_position
        self.hover_position = self.reset_hover_position()
        self.prev_pos = np.array([[0.0, 0.0, 0.5]])

    def reset_hover_position(self):
        if self.fixed_hover_position is not None:
            return self.fixed_hover_position
        else:
            new_hover_position = np.array([
                -1 + np.random.rand() * 2, # x -> range [-1, 1]
                -1 + np.random.rand() * 2, # y -> range [-1, 1]
                0  + np.random.rand() * 1, # z -> range [0,  2]
            ])
            return new_hover_position

    def reset(self, seed=None, options=dict()):
        """reset.

        Args:
            seed: seed to pass to the base environment.
            options: None
        """
        super().begin_reset(seed, options)
        self.hover_position = self.reset_hover_position()
        super().end_reset(seed, options)

        return self.state, self.info

    def compute_state(self):
        """Computes the state of the current timestep.

        This returns the observation.
        - ang_vel (vector of 3 values)
        - ang_pos (vector of 3/4 values)
        - lin_vel (vector of 3 values)
        - lin_pos (vector of 3 values)
        - previous_action (vector of 4 values)
        - hover_pos (vector of 3 values)
        """
        if self.state is not None: 
            self.prev_pos = self.state[10:13]

        ang_vel, ang_pos, lin_vel, lin_pos, quarternion = super().compute_attitude()

        # combine everything
        if self.angle_representation == 0:
            self.state = np.array(
                [*ang_vel, *ang_pos, *lin_vel, *lin_pos, *self.action, *self.hover_position]
            )
        elif self.angle_representation == 1:
            self.state = np.array(
                [*ang_vel, *quarternion, *lin_vel, *lin_pos, *self.action, *self.hover_position]
            )

    def compute_base_term_trunc_reward(self) -> None:
        """Overwrite the base class reward function to give credit for crashing the drone to the attacker."""
        # exceed step count
        if self.step_count > self.max_steps:
            self.truncation |= True
        # collision
        if np.any(self.env.contact_array):
            self.reward = +100.0            # crashing is actually good for the attacker
            self.info["collision"] = True
            self.termination |= True
        # exceed flight dome
        if np.linalg.norm(self.env.state(0)[-1]) > self.flight_dome_size:
            self.reward = -100.0
            self.info["out_of_bounds"] = True
            self.termination |= True

    def compute_term_trunc_reward(self):
        """Computes the termination, truncation, and reward of the current timestep."""
        self.compute_base_term_trunc_reward()

        if not self.sparse_reward:

            dist_to_hover = np.linalg.norm(self.env.state(0)[-1] - self.state[-3:])
            dist_to_prev_pos = np.linalg.norm(self.env.state(0)[-1] - self.prev_pos)
            
            ang_vel, lin_vel = np.linalg.norm(self.env.state(0)[0]), np.linalg.norm(self.env.state(0)[2])
            ang_dist = np.linalg.norm(self.env.state(0)[1][:2])

            action_cost = sum([max(i, 0) for i in self.action])

            self.reward += (dist_to_hover*1.5 + ang_dist)
            self.reward -= action_cost
            self.reward -= 1.5
    
