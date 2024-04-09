import torch, wandb, itertools
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

# Prepare model configuration
config = {
    "policy_type": 'MlpPolicy',

    "total_training_timesteps": 1_600_000,
    "checkpoint_frequency": (150_000//3)*3,  # This must by a multiple of the number of parallel environments
    "eval_frequency": (30_000//3)*3,         # This must by a multiple of the number of parallel environments
    "nb_eval_episodes": 20,

    "spawn_point": None,
    "eval_hover_point": [0.0, -1.0, 2.0],

    "model": 'Agilicious',
    "save_path": f'models/nominal/runs/{datetime.now().strftime("%Y-%m-%d/%H-%M-%S")}',
    "nb_parallel_env": 3,
}

# Generate all combinations of hyperparameters
H = {
    'architecture' : [[64, 64], [128, 64], [128, 128]],
    'learning_rate': [5e-5, 3e-4, 1e-4],
    'batch_size'   : [96, 192],
    'gamma'        : [0.95], # 0.99
    'n_steps'      : [2_560, 5_120],
}

keys, values = zip(*H.items())
H = [dict(zip(keys, v)) for v in itertools.product(*values)]

# ----- Hyperparameter tuning -----
for hyperparam in H:
    
    config['hyperparameters'] = hyperparam

    # Add run's information to config
    config['run_name'] = f"NOM_arch={'A' if sum(hyperparam['architecture']) == 128 else 'B' if sum(hyperparam['architecture']) == 128+64 else 'C'}, lr={round(hyperparam['learning_rate'],6)}, bs={hyperparam['batch_size']}, gamma=.95, T={hyperparam['n_steps']}"

    # Define W&B (logging system) run 
    run = wandb.init(
        project="ROBOTO-AGIL",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        tags=['Nominal', config['model']],
        name=config['run_name'],
    )

    # Save config file
    config['run_id'], config['run_url'] = run.id, run.url
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
        n_steps=hyperparams['n_steps'],
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
                hover_point=config['eval_hover_point'],
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
