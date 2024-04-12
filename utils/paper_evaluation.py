import subprocess, os

MODELS = {
    # 'nominal':  'checkpoints/paper_results/model-1_899K-iter-NOM',
    # 'attacker': 'checkpoints/paper_results/model-2_012K-iter-ATT',
    # 'defender': 'checkpoints/paper_results/model-3_010K-iter-DEF',
}

HOVER_POINTS = [
    [0.85, 0.90, 1.7],
    [0.0, 0.0, 0.5],
    [0.0, 0.0, 1.2],
    [0.7, 0.85, 0.7],
    [0.0, -1.0, 1.5],
    [-1.0, -1.0, 0.5],
]

NB_EVAL_EPISODES = 20


for model_type, model_path in MODELS.items():
    for POINT in HOVER_POINTS:
        eval_name = f"_{POINT[0]}_{POINT[1]}_{POINT[2]}_"
        os.makedirs(model_path + '/evaluation/' + eval_name, exist_ok=True)
        subprocess.run([
            'python3', '-m', f'models.{model_type}.evaluation.evaluate',
            '--model_path', model_path + '/model',
            '--hover_point', *[str(round(c, 1)) for c in POINT],
            '--nb_eval_episodes', str(NB_EVAL_EPISODES),
            '--eval_name', str(eval_name),
            '--disable_visual_eval',
        ])