# BaBee: An Autonomous Quadrotor for Agilicious-based Agents using Reinforcement Learning.
This project aims to autonomously flight an Agilicious-based agent using Reinforcement Learning. The physical properties of the quadrotor can be found (and changed) in `/UAV`, and pre-trained models in `/checkpoints`.

![Agilicious-based Agent Flying](docs/imgs/agilicious-flight.gif)

Was produced: a Nominal model to flight the quadrotor, an Attacker model to make an optimal attack on the quadrotor's sensors as a Man-in-the-Middle, and a Defender model to provide an optimal countermeasure on the quadrotor's attack.


## Setup
In order to set up your local Python environment, simply pip install the required libraries by running:

```pip install -r requirements.txt```

For more documentation on the physical quadrotor used (Agilicious), how to assemble it, set it up, etc; please refer to `/docs`.


## Nominal Model
The nominal model is the base agent responsible for taking actions in the environment. The nominal does not know whether it is under attack and therefore takes deterministic actions.

### Training
First, in order to set up the training and define parameters such as training time, evaluation settings, model hyperparameters, learning algorithms, number of parallel environments, etc; you must refer to ```nominal/train.py```.
We recommend only changing the parameters within the ```config```dictionary in that file.

Then, to train a nominal agent, you must be in the root path of the project (`/babee`) and start training as follows:

```python3 -m models.nominal.train```

While training, your model will be evaluated every X timesteps (results displayed on terminal), saved every Y timesteps (saved in `nominal/runs/your_run`), and tested every Y timesteps (with performance metrics plotted in `nominal/runs/your_run`).
X and Y variables can be set within ```nominal/train.py``` under ```config```.

Note that, additional training and evaluation performance metrics are reported to W&B (wandb - Weight&Biases) for convenience. Therefore, you may need to add your account key in `nominal/train.py` under ```wandb.login(...)```, if you wish to see those.

### Evaluation 
If you wish to evaluate a trained model without going through the training pipeline, you can run the following command:

```python3 -m models.nominal.evaluation.evaluate -p path/to/model -e 20 -hp 1 1 1``` 

where you can select from the below options:
 - -p (--model_path): absolute path to model to evaluate (required);
 - -e (--nb_eval_episodes): number of episodes per evaluation (required);
 - -hp (--hover_point): coordinates of the hover point formatted as '0 -1 2' (required);
 - -dv (--disable_visual_eval): flag to disable visualisation of the environment during evaluation.

(do not include '.zip' extension for the model, and make sure the model is from a training checkpoint such as `models/nominal/runs/2024-03-14/23-44-07/model-99K-iter/model` for example.)


## Attacker Model
The attacker model is the agent trained to make an optimal attack on the quadrotor's sensors. It is trained to crash the Nominal model as quick as possible, with as little energy cost as possible.

### Training & Evaluation
The training of the Attacker model is done in a very similar way as the Nominal. The only two differences are:
 1. For all absolute paths, replace `nominal` with `attacker`.
 2. You must first train a nominal model before training/evaluating an attacker. That is because the Attacker is learned on top of the Nominal. For that, you must either include a nominal model in `/checkpoints` as `/checkpoints/nominal_model.zip`; or include such a model anywhere else in the repository and change the path in `models/attacker/learning/rollout_collection.py` accordingly.


## Defender Model
The defender model is the agent trained to provide an optimal countermeasure on the quadrotor's attack. It is trained to maintain the stability of the Nominal under attack, with as little energy cost as possible.

### Training & Evaluation
The training of the Defender model is done in a very similar way as the Nominal. The two only differences are:
 1. For all absolute paths, replace `nominal` with `defender`.
 2. You must first train nominal and attacker models before training/evaluating a defender. That is because the Defender is learned on top of the Nominal and the Attacker. For that, you must either include those models in `/checkpoints` as `/checkpoints/nominal_model.zip` and `/checkpoints/attacker_model.zip`; or include such models anywhere else in the repository and change the paths in `models/defender/learning/rollout_collection.py` accordingly.

## Thank you
Feel free to use the source code for any research purpose.

### If you use this repository, please cite:
```
@journal{belkadi2024securecontrolsystems,
      title={Secure Control Systems for Autonomous Quadrotors against Cyber-Attacks}, 
      author={Samuel Belkadi},
      year={2024},
}
```
