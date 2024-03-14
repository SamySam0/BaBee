""" 
Base Environment for the QuadX model using PyFlyt and Gymnasim.

This module defines the base environment for the QuadX model using PyFlyt and Gymnasim. 
It provides the necessary functionality and interfaces for interacting with the environment, 
including resetting the environment, stepping through the environment, and rendering the environment.

Classes:
- QuadXBaseEnv: Base environment class for the QuadX model.

"""

import gymnasium, pybullet
import numpy as np

from typing import Any
from gymnasium import spaces
from PyFlyt.core.aviary import Aviary
from UAV.agilicious import Agilicious
from PyFlyt.core.utils.compile_helpers import check_numpy


class QuadXBaseEnv(gymnasium.Env):
    """ Base Environment for the QuadX model using PyFlyt and Gymnasim. """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 120}

    def __init__(
        self,
        start_pos: np.ndarray = np.array([[0.0, 0.0, 1.0]]), 
        start_orn: np.ndarray = np.array([[0.0, 0.0, 0.0]]),
        flight_dome_size: float = np.inf,
        max_duration_seconds: float = 10.0,
        angle_representation: str = "quaternion",
        agent_hz: int = 40,
        render_mode: None | str = None,
        render_resolution: tuple[int, int] = (480, 480),
    ):
        """ 
        Args:
            start_pos (np.ndarray): start_pos
            start_orn (np.ndarray): start_orn
            flight_dome_size (float): flight_dome_size
            max_duration_seconds (float): max_duration_seconds
            angle_representation (str): angle_representation
            agent_hz (int): agent_hz
            render_mode (None | str): render_mode
            render_resolution (tuple[int, int]): render_resolution
        """

        if 120 % agent_hz != 0:
            raise AssertionError(f"`agent_hz` must be round denominator of 120.")

        if render_mode is not None:
            assert (render_mode in self.metadata["render_modes"]), \
                f"Invalid render mode {render_mode}, only {self.metadata['render_modes']} allowed."
        
        self.render_mode = render_mode
        self.render_resolution = render_resolution


        """ GYMNASIUM CONSTANTS """
        # Attitude size increases by 1 for quaternion
        if angle_representation == "euler":
            attitude_shape = 12
        elif angle_representation == "quaternion":
            attitude_shape = 13
        else:
            raise AssertionError(
                f"angle_representation must be either `euler` or `quaternion`, not {angle_representation}"
            )

        self.attitude_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(attitude_shape,), dtype=np.float64
        )
        self.target_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
        )
        
        angular_rate_limit = np.pi/3
        thrust_limit = 8.5*4 # *4 for collective thrust

        high = np.array(
            [
                angular_rate_limit,
                angular_rate_limit,
                angular_rate_limit,
                thrust_limit,
            ]
        )
        low = np.array(
            [
                -angular_rate_limit,
                -angular_rate_limit,
                -angular_rate_limit,
                0.0,
            ]
        )

        self.action_space = spaces.Box(low=low, high=high, dtype=np.float64)

        # The whole implicit state space = attitude + previous action + target location
        self.combined_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                attitude_shape
                + self.action_space.shape[0]
                + self.target_space.shape[0],
            ),
            dtype=np.float64,
        )

        """ ENVIRONMENT CONSTANTS """
        self.start_pos = start_pos
        self.start_orn = start_orn
        self.flight_dome_size = flight_dome_size
        self.max_steps = int(agent_hz * max_duration_seconds)
        self.env_step_ratio = int(120 / agent_hz)
        if angle_representation == "euler":
            self.angle_representation = 0
        elif angle_representation == "quaternion":
            self.angle_representation = 1

    def close(self) -> None:
        """ Disconnects the internal Aviary. """
        # If we already have an env, disconnect from it
        if hasattr(self, "env"):
            self.env.disconnect()

    def reset(
        self, seed: None | int = None, options: dict[str, Any] = dict()
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Args:
            seed (None | int): seed
            options (dict[str, Any]): options

        Returns:
            tuple[dict[str, Any], dict[str, Any]]:
        """
        raise NotImplementedError

    def begin_reset(self, seed=None, options=dict(), drone_options=dict()) -> None:
        """The first half of the reset function."""
        super().reset(seed=seed)

        # if we already have an env, disconnect from it
        if hasattr(self, "env"):
            self.env.disconnect()

        self.step_count = 0
        self.termination = False
        self.truncation = False
        self.state = None
        self.action = np.zeros((4,))
        self.reward = 0.0
        self.info = {}
        self.info["out_of_bounds"] = False
        self.info["collision"] = False
        self.info["env_complete"] = False

        drone_type_mappings = dict() 
        drone_type_mappings["agilicious"] = Agilicious

        # init env
        self.env = Aviary(
            start_pos=self.start_pos,
            start_orn=self.start_orn,
            drone_type_mappings=drone_type_mappings, 
            drone_type="agilicious", 
            render=self.render_mode is not None,
            # drone_options=drone_options,
            seed=seed,
        )

        if self.render_mode is not None:
            self.camera_parameters = self.env.getDebugVisualizerCamera()

    def end_reset(
        self, seed: None | int = None, options: dict[str, Any] = dict()
    ) -> None:
        """The tailing half of the reset function."""
        # register all new collision bodies
        self.env.register_all_new_bodies()

        # set flight mode
        self.env.set_mode(0)

        # wait for env to stabilize
        for _ in range(10):
            self.env.step()

        self.compute_state()

    def compute_state(self) -> None:
        """Computes the state of the QuadX."""
        raise NotImplementedError

    def compute_auxiliary(self) -> np.ndarray:
        """This returns the auxiliary state form the drone."""
        return self.env.aux_state(0)

    def compute_attitude(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """state.

        This returns the base attitude for the drone.
        - ang_vel (vector of 3 values)
        - ang_pos (vector of 3 values)
        - lin_vel (vector of 3 values)
        - lin_pos (vector of 3 values)
        - quarternion (vector of 4 values)
        """
        raw_state = self.env.state(0)

        # state breakdown
        ang_vel = raw_state[0]
        ang_pos = raw_state[1]
        lin_vel = raw_state[2]
        lin_pos = raw_state[3]

        # quarternion angles
        quarternion = pybullet.getQuaternionFromEuler(ang_pos)

        return ang_vel, ang_pos, lin_vel, lin_pos, quarternion

    def compute_term_trunc_reward(self) -> None:
        """compute_term_trunc_reward."""
        raise NotImplementedError

    def compute_base_term_trunc_reward(self) -> None:
        """compute_base_term_trunc_reward."""
        # exceed step count
        if self.step_count > self.max_steps:
            self.truncation |= True

        # collision
        if np.any(self.env.contact_array):
            self.reward = -200.0
            self.info["collision"] = True
            self.termination |= True

        # exceed flight dome
        if np.linalg.norm(self.env.state(0)[-1]) > self.flight_dome_size:
            self.reward = -100.0
            self.info["out_of_bounds"] = True
            self.termination |= True

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Steps the environment.

        Args:
            action (np.ndarray): action

        Returns:
            state, reward, termination, truncation, info
        """
        # unsqueeze the action to be usable in aviary
        self.action = action.copy()

        # reset the reward and set the action
        self.reward = -0.1
        self.env.set_setpoint(0, action)

        # step through env, the internal env updates a few steps before the outer env
        for _ in range(self.env_step_ratio):
            # if we've already ended, don't continue
            if self.termination or self.truncation:
                break

            self.env.step()

            # compute state and done
            self.compute_state()
            self.compute_term_trunc_reward()

        # increment step count
        self.step_count += 1

        return self.state, self.reward, self.termination, self.truncation, self.info

    def render(self) -> np.ndarray:
        """render."""
        check_numpy()
        assert (
            self.render_mode is not None
        ), "Please set `render_mode='human'` or `render_mode='rgb_array'` to use this function."

        _, _, rgbaImg, _, _ = self.env.getCameraImage(
            width=self.render_resolution[1],
            height=self.render_resolution[0],
            viewMatrix=self.camera_parameters[2],
            projectionMatrix=self.camera_parameters[3],
        )

        rgbaImg = np.asarray(rgbaImg).reshape(
            self.render_resolution[0], self.render_resolution[1], -1
        )

        return rgbaImg