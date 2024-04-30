from helper_classes import *
from Transformer.transformer_trainer import *
from HPO_RL import *
import os

max_layers, batch_size= 8, 16
size_buffer = batch_size * 30
train_loader = torch.utils.data.DataLoader(

            torchvision.datasets.MNIST('/files/', train=True, download = True, 
                                       transform = torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,)
                                           )
                                       ])),
            batch_size = batch_size, shuffle=False
        )

test_loader = torch.utils.data.DataLoader(

            torchvision.datasets.MNIST('/files/', train=False, download = True, 
                                    transform = torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,)
                                        )
                                    ])),
            batch_size = batch_size, shuffle=False
        )

experiment_nb = 1
rlhpo = RLHPO(max_layers=max_layers, experiment_number=experiment_nb)
rlhpo.train_loader = train_loader
rlhpo.test_loader = test_loader
rlhpo.is_testing = True

explainability_dir = f"{RESULTS_DIR}/exp{experiment_nb}/explainability_3"
os.makedirs(explainability_dir, exist_ok=True)

for run_nbr in range(1, 6):
    df_res = pd.DataFrame()
    df_layers = pd.DataFrame()
    # for i  in range(950, 1000, 5):
    for i  in range(920, 990, 10):
        state_encoder = StateEncoding(action_space= 4, perf_space=32, output_layer=64)
        state_encoder.load_state_dict(torch.load(f'{MODELS_DIR}/exp{experiment_nb}/EP-{i}_state_encoder.pt'))
        state_encoder.eval()

        transformer_trainer = TransformerTrainer(max_layers, 64, num_layers=2, 
            expansion_factor=4, n_heads=4, action_space=4, size_buffer = size_buffer,
            env = rlhpo, target_episode = 75, state_encoder = state_encoder, training_loader=train_loader, 
            testing_loader=test_loader, saving_dir=f"{RESULTS_DIR}/exp{experiment_nb}")

        transformer_trainer.eval()
        transformer_trainer.load_models(f'{MODELS_DIR}/exp{experiment_nb}/EP-{i}')
        df_ep_res, df_layer = rlhpo.interpretability(i, transformer_trainer)
        df_res = pd.concat([df_res, df_ep_res])
        df_layers = pd.concat([df_layers, df_layer])


    df_res.to_csv(f"{explainability_dir}/res_{run_nbr}.csv")
    df_layers.to_csv(f"{explainability_dir}/layers_{run_nbr}.csv")



