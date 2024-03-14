import torch
from stable_baselines3.common.on_policy_algorithm import *
from stable_baselines3 import PPO

# Pre-trained models
nominal = PPO.load('checkpoints/nominal_model')

# Rollout collection
def rollout_collection(self, env, callback, rollout_buffer, n_rollout_steps, nominal=nominal):
    assert self._last_obs is not None, "No previous observation was provided"

    # Switch to eval mode (this affects batch norm / dropout)
    self.policy.set_training_mode(False)

    n_steps = 0
    rollout_buffer.reset()
    # Sample new weights for the state dependent exploration
    if self.use_sde:
        self.policy.reset_noise(env.num_envs)

    callback.on_rollout_start()

    while n_steps < n_rollout_steps:
        if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
            # Sample a new noise matrix
            self.policy.reset_noise(env.num_envs)

        with torch.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(self._last_obs, self.device)

            actions_nominal, _, _ = nominal.policy(obs_tensor) 
            actions_attacker, values_attacker, log_probs_attacker = self.policy(obs_tensor) 

            combined_actions = actions_nominal + actions_attacker

        combined_actions = combined_actions.cpu().numpy() 
        actions_attacker = actions_attacker.cpu().numpy() 

        # Rescale and perform action
        clipped_actions = combined_actions 

        if isinstance(self.action_space, spaces.Box):
            if self.policy.squash_output:
                # Unscale the actions to match env bounds
                # if they were previously squashed (scaled in [-1, 1])
                clipped_actions = self.policy.unscale_action(clipped_actions)
            else:
                # Otherwise, clip the actions to avoid out of bound error
                # as we are sampling from an unbounded Gaussian distribution
                clipped_actions = np.clip(combined_actions, self.action_space.low, self.action_space.high)

        new_obs, rewards, dones, infos = env.step(clipped_actions)

        self.num_timesteps += env.num_envs

        # Give access to local variables
        callback.update_locals(locals())
        if not callback.on_step():
            return False
        
        self._update_info_buffer(infos)
        n_steps += 1
        
        if isinstance(self.action_space, spaces.Discrete):
            # Reshape in case of discrete action
            actions_attacker = actions_attacker.reshape(-1, 1) 
        
        # Handle timeout by bootstraping with value function
        # see GitHub issue #633
        for idx, done in enumerate(dones):
            if (
                done
                and infos[idx].get("terminal_observation") is not None
                and infos[idx].get("TimeLimit.truncated", False)
            ):
                terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                with torch.no_grad():
                    terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                rewards[idx] += self.gamma * terminal_value

        rollout_buffer.add(
            self._last_obs,  # type: ignore[arg-type]
            actions_attacker,  
            rewards,
            self._last_episode_starts,  # type: ignore[arg-type]
            values_attacker,
            log_probs_attacker,
        )
        self._last_obs = new_obs  # type: ignore[assignment]
        self._last_episode_starts = dones

    with torch.no_grad():
        # Compute value for the last timestep
        values_attacker = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

    rollout_buffer.compute_returns_and_advantage(last_values=values_attacker, dones=dones)

    callback.update_locals(locals())

    callback.on_rollout_end()
    return True

OnPolicyAlgorithm.collect_rollouts = rollout_collection
