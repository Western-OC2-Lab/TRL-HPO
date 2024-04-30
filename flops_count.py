from helper_classes import *
from Transformer.transformer_trainer import *
from HPO_RL import *
import os
from calflops import calculate_flops

max_layers, batch_size= 6, 16
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
i = 0
state_encoder = StateEncoding(action_space= 4, perf_space=32, output_layer=64)
state_encoder.load_state_dict(torch.load(f'{MODELS_DIR}/exp{experiment_nb}/EP-{i}_state_encoder.pt'))
state_encoder.eval()

transformer_trainer = TransformerTrainer(max_layers, 64, num_layers=2, 
    expansion_factor=4, n_heads=4, action_space=4, size_buffer = size_buffer,
    env = rlhpo, target_episode = 75, state_encoder = state_encoder, training_loader=train_loader, 
    testing_loader=test_loader, saving_dir=f"{RESULTS_DIR}/exp{experiment_nb}")

transformer_trainer.eval()
transformer_trainer.load_models(f'{MODELS_DIR}/exp{experiment_nb}/EP-{i}')
# transformer_trainer.eval()
# rlhpo.eval(i, state_encoder, transformer_trainer)

actor= transformer_trainer.actor
input_size = (1, 6, 64)
flops, macs, params = calculate_flops(model=actor, 
                                      input_shape=input_size,
                                      output_as_string=True,
                                      output_precision=4)
print("Alexnet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))