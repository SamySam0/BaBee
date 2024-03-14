import numpy as np
from utils.graph_plotting import plot_distance_to_hover, plot_position_in_3d_space


# Evaluation script for visualisation
def visual_evaluation(model, eval_env, n_eval_episodes):
    obs, _ = eval_env.reset()

    # Simple simulation for visual analysis
    run_id = 0
    while run_id < n_eval_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, _ = eval_env.step(action)
        if done or trunc:
            obs, _ = eval_env.reset()
            run_id += 1


# Get coordinates of drone at each timestep over a single simulation
def get_results_of_single_sim(model, eval_env, deterministic=False):
    xs, ys, zs = [], [], []
    obs, _ = eval_env.reset()

    # Launch a single simulation (until terminates)
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, done, trunc, _ = eval_env.step(action)
        lin_pos = obs[10:13]

        # Record X, Y and Z coordinates
        xs.append(lin_pos[0])
        ys.append(lin_pos[1])
        zs.append(lin_pos[2])

        if done or trunc:
            obs, _ = eval_env.reset()
            break
    
    return xs, ys, zs


# Evaluation script for plotting in 2D (euclidean distance to hover point over time)
def evaluate_to_plot_2d(model, eval_env, n_eval_episodes, hover_point, plot_path):
    all_errors_to_hover = np.zeros((n_eval_episodes, 402*3)) # *3 for 120Hz

    # Save average error to hover point over time
    for run_id in range(n_eval_episodes):

        # Get X,Y,Z positions at each timestep
        xs, ys, zs = get_results_of_single_sim(model, eval_env)
        coordinates = np.stack((xs, ys, zs), axis=-1)

        # Calculate ditance to hover point for each timestep
        errors = np.linalg.norm(coordinates - hover_point, axis=1)
        errors = np.pad(errors, (0, (402*3)-len(errors)), 'edge') # Fill in the episode length
        all_errors_to_hover[run_id] = errors

    # Plot euclidean distance over time
    plot_distance_to_hover(
        save_path = f'{plot_path}/2D-Euclidean_Dist.png',
        title = f"Distance of quadrotor to hover point through time averaged over {n_eval_episodes} runs.", 
        errors = np.mean(all_errors_to_hover, axis=0),
        std = np.std(all_errors_to_hover, axis=0),
    )


# Evaluation script for plotting in 3D (position of the drone in 3D space over time)
def evaluate_to_plot_3d(model, eval_env, hover_point, plot_path):
    
    # Run one simulation and get X,Y,Z positions at each timestep
    xs, ys, zs = get_results_of_single_sim(model, eval_env)

    # Plot 3D graphs
    for graph_type in ['ZOOM', 'FIT_PATH', 'FULL_WORLD']:
        plot_position_in_3d_space(
            xs, ys, zs,
            save_path = f'{plot_path}/3D-{graph_type}.png',
            hover_point = hover_point,
            graph_type = graph_type,
        )

