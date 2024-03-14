import torch, wandb
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO, A2C, SAC

from utils.logging import save_config_file

from models.nominal.env.hover_env import QuadXHoverAgiliciousNomEnv
from utils.callbacks import ModelCheckpointCallback, ModelEvalCallback, ModelTrainingCallback
from stable_baselines3.common.callbacks import CallbackList

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed


# ----- Setup -----

# W&B Run Logging
wandb.login(key='7425c67d3c5151a3744fe900a66cc0a3850c0858')

# # Prepare model configuration
# config = {
#     "policy_type": 'MlpPolicy',

#     "total_training_timesteps": 2_000_000,
#     "checkpoint_frequency": 200_000*3,
#     "eval_frequency": 50_000*3,
#     "nb_eval_episodes": 20,

#     "spawn_point": None,
#     "eval_hover_point": [0.0, -1.0, 2.0],

#     "model": 'Agilicious',
#     "save_path": f'runs/{datetime.now().strftime("%Y-%m-%d/%H-%M-%S")}',
#     "nb_parallel_env": 3,

#     "hyperparameters": {
#         'architecture': [512, 512, 256, 128],
#         'learning_rate': 3e-4, # 1e-3?
#         'batch_size'   : 1024,
#         'gamma'        : 0.95, # 0.99 ?
#         'n_steps'      : 2_000,
#     },
# }

import itertools

H = {
    'architecture' : [[64, 64], [128, 64], [256, 128]],
    'learning_rate': [5e-5, 3e-4, 1e-4],
    'batch_size'   : [64, 128],
    'gamma'        : [0.99], # 0.95
    'n_steps'      : [1_024, 2_560, 6_400],
}
H = {
    'architecture' : [[128, 64]],
    'learning_rate': [3e-4],
    'batch_size'   : [128],
    'gamma'        : [0.99], # 0.95
    'n_steps'      : [2_560],
}

# Generate all combinations of parameters
keys, values = zip(*H.items())
H = [dict(zip(keys, v)) for v in itertools.product(*values)]

for h in H:

    config = {
        "policy_type": 'MlpPolicy',

        "total_training_timesteps": 2_000_000,
        "checkpoint_frequency": 200_000*3,
        "eval_frequency": 50_000*3,
        "nb_eval_episodes": 20,

        "spawn_point": None,
        "eval_hover_point": [0.0, -1.0, 2.0],

        "model": 'Agilicious',
        "save_path": f'runs/{datetime.now().strftime("%Y-%m-%d/%H-%M-%S")}',
        "nb_parallel_env": 3,
    }

    config['hyperparameters'] = h



    # Define W&B (logging system) run 
    # wandb.tensorboard.patch(root_logdir=config['save_path'])
    run = wandb.init(
        project="ROBOTO",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
    )

    # Add run's information to config
    config['run_id'], config['run_name'], config['run_url'] = run.id, run.name, run.url

    # Save config file
    save_config_file(config, path=config['save_path'])


    # ----- Parallel Environment -----
    def make_env(rank, seed=0):
        def _init():
            env = Monitor(QuadXHoverAgiliciousNomEnv(sparse_reward=False)) # Wrap in Monitor to get training metrics
            env.reset(seed=seed + rank)
            return env

        set_random_seed(seed) # Used for enhanced reproducibility
        return _init

    # Create parallel environments
    n_procs = config['nb_parallel_env']
    parallel_env = SubprocVecEnv(
        [make_env(i + n_procs) for i in range(n_procs)],
        start_method="fork",
    )


    # ----- Training -----
    parallel_env.reset()

    # Load the model to train
    hyperparams = config['hyperparameters']
    model = PPO(   
        config["policy_type"], parallel_env, tensorboard_log=config['save_path'], verbose=1, device="cpu",
        policy_kwargs= None if hyperparams['architecture'] is None 
                            else dict(activation_fn=torch.nn.ReLU, net_arch=hyperparams['architecture']),
        learning_rate=hyperparams['learning_rate'],
        batch_size=hyperparams['batch_size'],
        gamma=hyperparams['gamma'],         
        n_steps=hyperparams['n_steps'], #//config['nb_parallel_env'],
    )

    # Load evaluation environment
    eval_env = Monitor(QuadXHoverAgiliciousNomEnv(
        sparse_reward=False, 
        fixed_hover_position=np.array(config['eval_hover_point']),
    ))

    # Train the model
    model.learn(
        total_timesteps=config['total_training_timesteps'],
        callback=CallbackList([
            ModelCheckpointCallback(
                save_freq=config['checkpoint_frequency'], 
                save_path=config['save_path'],
                nb_eval_episodes=config['nb_eval_episodes'],
            ),
            ModelEvalCallback(
                eval_env, eval_freq=config['eval_frequency'],
                n_eval_episodes=config['nb_eval_episodes'],
                deterministic=False, render=False,
            ),
            ModelTrainingCallback(),
        ])
    )

    # Terminate environments
    parallel_env.close()
    eval_env.close()

    # Terminate W&B run
    run.finish()
