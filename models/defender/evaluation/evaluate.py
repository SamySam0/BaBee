import argparse
import numpy as np
from stable_baselines3 import PPO

from stable_baselines3.common.monitor import Monitor
from models.defender.env.hover_env import QuadXHoverAgiliciousDefEnv
from models.defender.evaluation.evaluators import evaluate_to_plot_2d, evaluate_to_plot_3d, visual_evaluation


# ----- EVALUATION -----

if __name__ == "__main__":

    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="Select options for evaluation stage.")
    parser.add_argument("-p", "--model_path", type=str, required=True, help="Path to model to evaluate.")
    parser.add_argument("-e", "--nb_eval_episodes", type=int, required=True, help="Number of episodes per evaluation.")
    parser.add_argument("-hp", "--hover_point", nargs=3, required=True, help="Coordinates of the hover point formatted as '0 -1 2'. ")
    parser.add_argument("-dv", "--disable_visual_eval", dest='visual_eval', action='store_false', help="Disables visualisation at each evaluation stage.")
    args = parser.parse_args()

    # Fetch parsed constants
    MODEL_PATH  = args.model_path
    EVAL_HOVER_POINT = np.array([round(float(c), 1) for c in args.hover_point])
    N_EVAL_EPISODES  = args.nb_eval_episodes
    VISUAL_EVAL = args.visual_eval

    # Fetch evaluation results path (same as model location)
    PLOT_SAVE_PATH = '/'.join(MODEL_PATH.split('/')[:-1]) + '/evaluation'

    # Load evaluation environments
    eval_env = Monitor(QuadXHoverAgiliciousDefEnv(
        sparse_reward=False, 
        fixed_hover_position=EVAL_HOVER_POINT,
    ))

    visual_eval_env = Monitor(QuadXHoverAgiliciousDefEnv(
        sparse_reward=False, 
        fixed_hover_position=EVAL_HOVER_POINT,
        render_mode="rgb_array",
    ))


    # Load models
    defender_model = PPO.load(MODEL_PATH, env=eval_env)
    attacker_model = PPO.load('checkpoints/attacker_model')
    nominal_model = PPO.load('checkpoints/nominal_model')

    # 2D Metric evaluation: Euclidean distance to hover point over time
    print(f'\n-- 2D Metric evaluation ({N_EVAL_EPISODES} runs) --')
    print('Description: Euclidean distance to hover point over time.')
    print('Hover point:', EVAL_HOVER_POINT)
    evaluate_to_plot_2d(
        nominal_model, attacker_model, defender_model, eval_env, 
        n_eval_episodes=N_EVAL_EPISODES, hover_point=EVAL_HOVER_POINT, plot_path=PLOT_SAVE_PATH)
    print('Plots saved in:', PLOT_SAVE_PATH)

    # 3D Metric evaluation: Position of the drone in 3D space over time
    print(f'\n-- 3D Metric evaluation (1 run) --')
    print('Description: Position of the drone in 3D space over time.')
    print('Hover point:', EVAL_HOVER_POINT)
    evaluate_to_plot_3d(nominal_model, attacker_model, defender_model, eval_env, hover_point=EVAL_HOVER_POINT, plot_path=PLOT_SAVE_PATH)
    print('Plots saved in:', PLOT_SAVE_PATH)

    # Visual simulation
    if VISUAL_EVAL:
        print('\n-- Visual simulation (2 runs) --')
        print('Hover point:', EVAL_HOVER_POINT)
        attacker_model.env = visual_eval_env
        visual_evaluation(nominal_model, attacker_model, defender_model, visual_eval_env, n_eval_episodes=2)
        print('Recording could not be saved.')

    # Terminate environments
    eval_env.close()
    visual_eval_env.close()
