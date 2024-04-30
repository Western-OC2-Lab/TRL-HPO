from HPO_RL import *
import os
from constants import *

if __name__ == '__main__':
    experiment_nb = 1

    os.makedirs(f"{MODELS_DIR}/exp{experiment_nb}", exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/exp{experiment_nb}", exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/exp{experiment_nb}/final_results", exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/exp{experiment_nb}/figures", exist_ok=True)

    total_episodes = 500
    rl_hpo = RLHPO(max_layers=6, target_episode= int(total_episodes * 0.8), experiment_number=experiment_nb)

    rl_hpo.train(total_episodes)