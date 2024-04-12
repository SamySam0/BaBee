import torch, copy
import numpy as np
from utils.graph_plotting import plot_distance_to_hover, plot_position_in_3d_space
from stable_baselines3.common.on_policy_algorithm import obs_as_tensor
from models.nominal.evaluation.evaluators import get_results_of_single_sim as get_results_of_single_sim_for_nominal


# Get merged and clip action from both nominal and attacker models
def get_merged_action(observation, nominal, attacker, step):
    # Transform observation nparray into tensor
    obs = obs_as_tensor(np.array([observation]), device='cpu')

    # Predict directly from model's policy network
    with torch.no_grad():
        action_a = attacker.policy._predict(obs, deterministic=False)   # Attacker should stay stochastic for evaluation
        action_n = nominal.policy._predict(obs, deterministic=True)     # However, Nominal model should be deterministic
    
    # Merge and clip actions
    if step > 120*2:
        action = action_n + action_a
    else:
        action = action_n
    action = np.clip(action.cpu().numpy(), attacker.action_space.low, attacker.action_space.high)[0]
    return action


# Evaluation script for visualisation
def visual_evaluation(nominal_model, attacker_model, eval_env, n_eval_episodes):
    obs, _ = eval_env.reset()

    # Simple simulation for visual analysis
    run_id, step = 0, 0
    while run_id < n_eval_episodes:
        action = get_merged_action(observation=obs, nominal=nominal_model, attacker=attacker_model, step=step)
        obs, _, done, trunc, _ = eval_env.step(action)
        step += 1
        if done or trunc:
            obs, _ = eval_env.reset()
            run_id += 1
            step = 0


# Get coordinates of drone at each timestep over a single simulation
def get_results_of_single_sim(nominal_model, attacker_model, eval_env):
    xs, ys, zs = [], [], []
    obs, _ = eval_env.reset()
    step = 0

    # Launch a single simulation (until terminates)
    while True:
        action = get_merged_action(observation=obs, nominal=nominal_model, attacker=attacker_model, step=step)
        obs, _, done, trunc, _ = eval_env.step(action)
        lin_pos = obs[10:13]
        step += 1

        # Record X, Y and Z coordinates
        xs.append(lin_pos[0])
        ys.append(lin_pos[1])
        zs.append(lin_pos[2])

        if done or trunc:
            obs, _ = eval_env.reset()
            step = 0
            break
    
    return xs, ys, zs


# Evaluation script for plotting in 2D (euclidean distance to hover point over time)
def evaluate_to_plot_2d(nominal_model, attacker_model, eval_env, n_eval_episodes, hover_point, plot_path):
    # Save errors when under attack and nominal only
    all_errors_to_hover_ua = np.zeros((n_eval_episodes, 402*3)) # *3 for 120Hz
    all_errors_to_hover_no = np.zeros((n_eval_episodes, 402*3)) # *3 for 120Hz

    # Save average error to hover point over time
    # For both nominal and attacker, for comparison
    for run_id in range(n_eval_episodes):

        # -- Under Attack --
        # Get X,Y,Z positions at each timestep
        xs, ys, zs = get_results_of_single_sim(nominal_model, attacker_model, eval_env)
        coordinates = np.stack((xs, ys, zs), axis=-1)

        # Calculate ditance to hover point for each timestep
        errors = np.linalg.norm(coordinates - hover_point, axis=1)
        errors = np.pad(errors, (0, (402*3)-len(errors)), 'edge') # Fill in the episode length
        all_errors_to_hover_ua[run_id] = errors

        # -- Nominal Only (reference) --
        # Get X,Y,Z positions at each timestep
        xs, ys, zs = get_results_of_single_sim_for_nominal(nominal_model, eval_env, deterministic=True)
        coordinates = np.stack((xs, ys, zs), axis=-1)

        # Calculate ditance to hover point for each timestep
        errors = np.linalg.norm(coordinates - hover_point, axis=1)
        errors = np.pad(errors, (0, (402*3)-len(errors)), 'edge') # Fill in the episode length
        all_errors_to_hover_no[run_id] = errors

    # Plot euclidean distance over time
    plot_distance_to_hover(
        save_path = f'{plot_path}/2D-Euclidean_Dist.png',
        title = f"Distance of quadrotor to hover point through time averaged over {n_eval_episodes} runs.", 
        errors = np.mean(all_errors_to_hover_ua, axis=0),
        std = np.std(all_errors_to_hover_ua, axis=0),
        reference = (
            np.mean(all_errors_to_hover_no, axis=0),
            np.std(all_errors_to_hover_no, axis=0),
        )
    )


# Evaluation script for plotting in 3D (position of the drone in 3D space over time)
def evaluate_to_plot_3d(nominal_model, attacker_model, eval_env, hover_point, plot_path):
    
    # Run one simulation and get X,Y,Z positions at each timestep
    xs_ua, ys_ua, zs_ua = get_results_of_single_sim(nominal_model, attacker_model, eval_env)
    xs_no, ys_no, zs_no = get_results_of_single_sim_for_nominal(nominal_model, eval_env, deterministic=True)

    # Plot 3D graphs
    for graph_type in ['ZOOM', 'FIT_PATH', 'FULL_WORLD']:
        plot_position_in_3d_space(
            xs_ua, ys_ua, zs_ua,
            reference = (xs_no, ys_no, zs_no),
            save_path = f'{plot_path}/3D-{graph_type}.png',
            hover_point = hover_point,
            graph_type = graph_type,
        )

