import subprocess, os
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Logger


class ModelCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, nb_eval_episodes, hover_point):
        super(ModelCheckpointCallback, self).__init__(verbose=0)
        self.save_freq = save_freq
        self.save_path = save_path
        self.nb_eval_episodes = nb_eval_episodes
        self.hover_point = hover_point

    def _on_training_end(self):
        if (self.num_timesteps % self.save_freq != 0):
            self._on_step(force=True)

    def _on_step(self, force=False):
        if (self.num_timesteps % self.save_freq == 0) or force:

            # Save model checkpoint
            model_type = self.save_path.split('/')[1] # nominal, attacker or defender
            epoch_save_path = f'{self.save_path}/model-{int(self.num_timesteps//1e3):_}K-iter'
            self.model.save(epoch_save_path + '/model')

            # Evaluate model
            os.makedirs(epoch_save_path + '/evaluation', exist_ok=True)
            subprocess.run(['python3', '-m', f'models.{model_type}.evaluation.evaluate',
                            '--model_path', epoch_save_path + '/model',
                            '--hover_point', *[str(round(c, 1)) for c in self.hover_point],
                            '--nb_eval_episodes', str(self.nb_eval_episodes),
                            '--disable_visual_eval'])
        return True


class ModelEvalCallback(EventCallback):
    def __init__(
            self, eval_env, n_eval_episodes, eval_freq,
            deterministic=False, render=False, verbose=1,
        ):
        super().__init__(verbose=verbose)

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.render = render
        self.eval_env = eval_env

        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []

    def _log_success_callback(self, locals_, globals_):
        info = locals_["info"]
        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_training_end(self):
        self._on_step(force=True)

    def _on_step(self, force=False):
        if (self.num_timesteps % self.eval_freq == 0) or force:

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                callback=self._log_success_callback,
            )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/std_reward", float(std_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            self.logger.record("eval/std_ep_length", float(std_ep_length))

            success_rate = np.mean(self._is_success_buffer) if len(self._is_success_buffer) > 0 else 0.0
            self.logger.record("eval/success_rate", float(success_rate))

            # Save evaluation results: helps if training has ended
            self.logger.dump()

        return True


class ModelTrainingCallback(BaseCallback):
    def __init__(self):
        super(ModelTrainingCallback, self).__init__(verbose=0)

    def safe_std(self, arr):
        return np.nan if len(arr) == 0 else float(np.std(arr))  # type: ignore[arg-type]

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        assert self.model.ep_info_buffer is not None

        # Record standard deviation of episode reward and length during training
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_std", self.safe_std([ep_info["r"] for ep_info in self.model.ep_info_buffer]))
            self.logger.record("rollout/ep_len_std", self.safe_std([ep_info["l"] for ep_info in self.model.ep_info_buffer]))
        