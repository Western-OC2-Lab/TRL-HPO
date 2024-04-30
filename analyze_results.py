from helper_classes import *
from Transformer.transformer_trainer import *
from HPO_RL import *
from tqdm import tqdm
import os
from constants import *

max_layers = 6
batch_size = 16
size_buffer = batch_size * 30
train_loader = torch.utils.data.DataLoader(

            torchvision.datasets.MNIST({MNIST_FILE_LOCAL}, train=True, download = True, 
                                       transform = torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,)
                                           )
                                       ])),
            batch_size = batch_size, shuffle=False
        )

test_loader = torch.utils.data.DataLoader(

            torchvision.datasets.MNIST({MNIST_FILE_LOCAL}, train=False, download = True, 
                                    transform = torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,)
                                        )
                                    ])),
            batch_size = batch_size, shuffle=False
        )

experiment_nb = 1
os.makedirs(f"{RESULTS_DIR}/exp{experiment_nb}", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/exp{experiment_nb}/final_results", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/exp{experiment_nb}/figures", exist_ok=True)
rlhpo = RLHPO(max_layers=max_layers, experiment_number=experiment_nb)
rlhpo.train_loader = train_loader
rlhpo.test_loader = test_loader
rlhpo.is_testing = True

for i in tqdm(np.arange(900,980, 10)):
    state_encoder = StateEncoding(action_space= 4, perf_space=32, output_layer=64)
    state_encoder.load_state_dict(torch.load(f'{MODELS_DIR}/exp{experiment_nb}/EP-{i}_state_encoder.pt'))
    state_encoder.eval()

    transformer_trainer = TransformerTrainer(max_layers, 64, num_layers=2, 
        expansion_factor=4, n_heads=4, action_space=4, size_buffer = size_buffer,
        env = rlhpo, target_episode = 75, state_encoder = state_encoder, training_loader=train_loader, 
        testing_loader=test_loader, saving_dir=f"{RESULTS_DIR}/exp{experiment_nb}")
    
    transformer_trainer.eval()

    transformer_trainer.load_models(f'{MODELS_DIR}/exp{experiment_nb}/EP-{i}')
    
    transformer_trainer.eval()
    rlhpo.eval(i, state_encoder, transformer_trainer)



