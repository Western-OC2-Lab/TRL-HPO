import torch
from functions import *
from helper_classes import *
from Transformer.transformer_trainer import TransformerTrainer
import torchvision
from tqdm import tqdm
import pandas as pd
import time
import random
from joblib import Parallel, cpu_count, delayed
from replay_memory import *
from constants import *


class RLHPO: 

    def __init__(self, max_layers, experiment_number, batch_size = 16, 
    size_state_space = 64, performace_pts = 32, threshold = 0.0001, target_episode = 21):
        self.size_state_space = size_state_space
        self.performace_pts = performace_pts
        self.exp_number = experiment_number
        self.max_layers = max_layers
        self.prev_layer, self.prev_layer_name, self.prev_nb_outputs = None, None, [1, 28, 28]
        self.prev_actions, self.prev_perf = torch.ones(size = (1, 4)) * (-1), torch.zeros(size = (1, self.performace_pts))

        self.batch_size = batch_size
        self.input_layer = torch.ones(size = (1, self.max_layers, self.size_state_space)) * (-1)
        self.all_prev_actions, self.all_rewards, self.all_perfs, self.all_layers = [], [], [], []

        self.size_buffer = 20000
        self.is_testing = False


        self.set_actions = torch.ones(size = (self.max_layers, 1, 4)) * (-1)
        self.threshold = threshold
        self.curr_arch = None
        self.eval_threshold = 0.6
        self.training_epochs = 5

        self.eval_improvement_thresh_episodes = 70
        self.best_eval = 0
        self.passed_episodes = 0
        self.last_best_episode = 0
        self.target_episode = 1*target_episode
        self.saving_dir = f"{MODELS_DIR}/exp{experiment_number}"

        self.set_results = {}
        self.invalid_model = False
        self.no_improv = False
        self.prev_arch = None

        if self.is_testing == False:
            self.init_datasets()

        self.init_encoder_transformer()
        self.select_values_validation = np.linspace(0, (self.training_epochs) * (len(self.validation_loader)//self.batch_size), self.performace_pts).astype(int)

        

    def init_encoder_transformer(self):
        self.state_encoder = StateEncoding(action_space= 4, perf_space=self.performace_pts, output_layer=self.size_state_space)
        self.state_encoder.apply(weights_init_uniform_rule)
        self.state_encoder.eval()
        self.transformer_trainer = TransformerTrainer(self.max_layers, self.size_state_space, num_layers=2, 
    expansion_factor=4, n_heads=4, action_space=4, size_buffer = self.size_buffer, env = self, target_episode = self.target_episode, state_encoder = self.state_encoder, training_loader=self.train_loader, testing_loader=self.test_loader, saving_dir=self.saving_dir)
        self.transformer_trainer.get_ready()

    def init_datasets(self):
        mnist_train_data = torchvision.datasets.MNIST(f"{MNIST_FILE_LOCAL}", train=True, download = True, 
                                       transform = torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,)
                                           )
                                       ]))
        
        train_size = 55000
        validation_size = 5000
        mnist_train, mnist_val = torch.utils.data.random_split(mnist_train_data, [train_size, validation_size])
        random_indices = np.random.choice(range(0, 55000), 20000)
        np.random.seed(1)
        subset_vals = torch.utils.data.Subset(mnist_train, random_indices)

        # self.train_loader = torch.utils.data.DataLoader(mnist_train, batch_size = self.batch_size, shuffle=True)
        self.train_loader = torch.utils.data.DataLoader(subset_vals, batch_size = self.batch_size, shuffle=True)

        self.validation_loader = torch.utils.data.DataLoader(mnist_val, batch_size = self.batch_size, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(

            torchvision.datasets.MNIST(f"{MNIST_FILE_LOCAL}", train=False, download = True, 
                                    transform = torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,)
                                        )
                                    ])),
            batch_size = self.batch_size, shuffle=True
        )

        test_iterator = iter(self.test_loader)
        img, _ = next(test_iterator)
        self.random_img = img[0]

    def build_architecture(self, set_actions, max_idx):
        if max_idx >= self.max_layers:
            return None, None
        initial_inputs, full_model = [1, 1, 28, 28], None
        prev_output = None
        acc_layers = None
        output_layer = None
        prev_layer_name = None
        layer = None
        for idx in range(max_idx+1):
            action = set_actions[idx][0].cpu()
            layer, layer_name, nb_neurons, _ = action_space_to_layer(initial_inputs, action, idx, prev_layer_name)
            if layer_name != "FCL":
                if acc_layers == None:
                    acc_layers = layer
                else:
                    acc_layers.append(layer[0])
                # print(acc_layers)
                # intr_model = nn.Sequential(*acc_layers)
                intr_model = acc_layers[-1]
                if prev_output == None:
                    random_input = self.random_img
                else:
                    random_input = prev_output
                # with torch.no_grad():
                try:
                    output_rand = intr_model(random_input)
                except RuntimeError:
                    return None, layer
                prev_output = output_rand

                # NC = prev_output.size(1) * prev_output.size(2) * prev_output.size(3)
                NC = prev_output.size(0) * prev_output.size(1) * prev_output.size(2)

                # initial_inputs = [output_rand.size(1), output_rand.size(2), output_rand.size(3)]
                initial_inputs = prev_output.size()
                output_layer = nn.Linear(NC, 10)

            else:
                if len(layer) > 1:
                    for l in layer:
                        acc_layers.append(l)
                initial_inputs = [nb_neurons, 1, 1]
            
                output_layer = nn.Linear(nb_neurons, 10)
            prev_layer_name = layer_name

        full_model = IntermediateClassifier(acc_layers, prev_layer_name, output_layer)
        self.curr_arch = full_model
        self.prev_layer_name = prev_layer_name

        return full_model, layer
    
    def new_model_definition(self, target_model):
        if self.prev_arch != None:
            nb_all_layers = len(self.prev_arch.all_layers)
            name_layer = "all_layers.{}.{}"
            set_all_eligible_layers = []
            for i in range(nb_all_layers):
                if hasattr(target_model.all_layers[i], "weight"):
                    target_model.all_layers[i].weight.data = self.prev_arch.all_layers[i].weight.data
                    set_all_eligible_layers.append(name_layer.format(i, "weight"))

                if hasattr(target_model.all_layers[i], "bias"):
                    classname = target_model.all_layers[i].__class__.__name__
                    if classname.find("Linear") != -1: 
                        target_model.all_layers[i].bias = self.prev_arch.all_layers[i].bias
                    else:
                        target_model.all_layers[i].bias.data = self.prev_arch.all_layers[i].bias.data

                    set_all_eligible_layers.append(name_layer.format(i, "bias"))

            for name, param in target_model.named_parameters():
                if name in set_all_eligible_layers:
                    param.requires_grad = False

        return target_model

    def train_new_model(self, model, optimizer):
        start_time = time.time()
        perf_arr = []
        
        model.train()
        model.cuda()
        epochs = 3 if self.is_testing else self.training_epochs

        for epoch in tqdm(range(epochs)):
            local_perf = []
            for idx, (img, label) in enumerate(self.train_loader):
                optimizer.zero_grad()
                img = img.cuda()
                output = model(img)
                criterion = nn.CrossEntropyLoss()
                loss_value = criterion(output.cpu().float(), label)
                loss_value.backward()
                optimizer.step()
                
                accuracy_batch = calculate_accuracy(output.cpu(), label)
                local_perf.append(accuracy_batch)
            print(f'Epoch Accuracy: {np.round(np.mean(local_perf), 3)}')
        end_time = time.time()
        total_train_time = round((end_time - start_time) / 60, 2)
        # print(f'training_time: {total_train_time}, per_arr:{np.average(perf_arr)}') 
        return perf_arr
    
    def validate_model(self, model):
        start_time = time.time()
        perf_arr = []
        model.cuda()
        model.eval()

        designated_loader = self.validation_loader if self.is_testing == False else self.test_loader

        for idx, (img, label) in enumerate(designated_loader):
            img = img.cuda()
            output = model(img)
            accuracy_batch = calculate_accuracy(output.cpu(), label)
            perf_arr.append(accuracy_batch)

        end_time = time.time()
        total_validation_time = round((end_time - start_time) / 60, 2)
        # print(f'validation_time: {total_validation_time}, perf_arr: {np.average(perf_arr)}')

        # return perf_arr[-32:]
        perf_arr = np.array(perf_arr)
        return perf_arr[self.select_values_validation]
    
    def reset_values(self):
        self.prev_layer, self.prev_layer_name, self.prev_nb_outputs = None, None, [1, 28, 28]
        # self.input_layer = torch.zeros(size = (1, self.max_layers, self.size_state_space))
        self.prev_actions, self.prev_perf = torch.ones(size = (1, 4)) * (-1), torch.zeros(size = (1, self.performace_pts))
        self.input_layer = torch.ones(size = (1, self.max_layers, self.size_state_space)) * (-1)

        self.idx = 0
        self.invalid_model = False
        self.no_improv = False
        self.curr_arch = None
        self.prev_arch = None
        self.prev_layer_name = None
        # self.set_actions = [self.prev_actions.numpy() for i in range(self.max_layers)]
        self.set_actions  = torch.ones(size = (self.max_layers, 1, 4)) * (-1)

    def parallelized_training(self, episode_nbr):
        curr_idx = 0
        start_time = time.time()
        transitions = []
        while curr_idx <= (self.max_layers - 1) and (self.invalid_model == False) and self.no_improv == False:
            self.input_layer = self.input_layer.cuda()
            self.idx = curr_idx
            state, idx, action, set_actions, reward, next_state, curr_perf, curr_acc, prev_layer, _, _, done = self.transformer_trainer.train(self.input_layer, episode_nbr)
            # transition = (state.cpu(), idx, action, set_actions, reward, next_state.cpu(), curr_perf, curr_acc, prev_layer, prev_arch, done)
            transition = (state.detach().cpu().numpy(), idx, action, set_actions, reward, next_state.detach().cpu().numpy(), curr_perf, curr_acc, prev_layer, done)

            transitions.append(transition)
            # self.rm.add(state.cpu(), idx, action, set_actions, reward, next_state.cpu(), curr_perf, curr_acc, prev_layer, prev_arch, done)
            
            curr_idx += 1
        self.reset_values()
        end_time = time.time()
        training_single_episode = round((end_time - start_time)/60, 4)
        # print('training_single_episode:', training_single_episode)

        return transitions


    def train(self, nb_episodes):
        
        for episode in tqdm(range(nb_episodes)):
            start_time = time.time()
            episode_optim = episode % 5 == 0
            # print('resources_before', torch.cuda.mem_get_info())
            start_time = time.time()
            transitions = Parallel(n_jobs=3, verbose=0)(delayed(self.parallelized_training)(s_episode) for s_episode in range(10))
            end_time = time.time()
            transition_time  = round((end_time - start_time)/60, 4)
            print(f'Training time of transitions (10 episodes): {transition_time}, nb transitions: {len(transitions)}')
            # print('resources_after', torch.cuda.mem_get_info())

            self.transformer_trainer.rm.add(transitions)

            if self.transformer_trainer.started_training and episode_optim:
                eval_res = self.eval(episode, self.state_encoder, self.transformer_trainer)
                shape_eval = eval_res.shape[0]-1
                reward_ep = eval_res.iloc[shape_eval]['reward']
                if reward_ep < 0:
                    perf_episode = eval_res.iloc[shape_eval-1]['perf']
                else:
                    perf_episode = eval_res.iloc[shape_eval]['perf']

                self.set_results[f"{episode}"] = perf_episode
                    
                if perf_episode > self.best_eval:
                    self.best_eval = perf_episode
                    self.last_best_episode = episode
                    self.passed_episodes = 0
                else:
                    self.passed_episodes += 5

                if self.passed_episodes == self.eval_improvement_thresh_episodes:
                    self.transformer_trainer.save_models(f"{nb_episodes}_validation")
                    df_results = pd.DataFrame.from_dict([self.set_results])
                    df_results.to_csv(f"{RESULTS_DIR}/exp{self.exp_number}/final_validation_results.csv")
                    break

            start_optimization_time = time.time()
            for i in tqdm(range(5)):
                self.transformer_trainer.optimize()
            end_optimization_time = time.time()

            opt_time = round((end_optimization_time - start_optimization_time)/60, 4)
            print('optimization_time', opt_time)

            self.reset_values()
            

            if episode % 10 == 0:
                if self.transformer_trainer.started_training:
                    self.transformer_trainer.save_models(episode)
                    if len(self.transformer_trainer.losses['critic_loss']) > 0:
                        df_results = pd.DataFrame()
                        df_results['critic_loss'] = self.transformer_trainer.losses['critic_loss']
                        df_results['actor_loss'] = self.transformer_trainer.losses['actor_loss']
                        df_results.to_csv(f"{RESULTS_DIR}/exp{self.exp_number}/all_results_{episode}.csv")

                        self.transformer_trainer.losses = {
                            'critic_loss': [],
                            'actor_loss': []
                        }
            end_time = time.time()
            episode_train_time = round((end_time - start_time) / 60, 2)
            print(f"Episode Train Time: {episode_train_time}")

        self.transformer_trainer.save_models(nb_episodes)
        # self.transformer_trainer.save_models(f"{nb_episodes}_validation")
        df_results = pd.DataFrame.from_dict([self.set_results])
        df_results.to_csv("final_validation_results.csv")

    def eval(self, episode, state_encoder, transformer_trainer):
        curr_idx = 0
        # state_encoder = state_encoder.cpu()
        # transformer_trainer.actor.cpu()
        all_prev_actions, all_rewards, all_perfs, all_layers, all_critic_eval = [], [], [], [], []
        while curr_idx <= (self.max_layers - 1) and (self.invalid_model == False) and self.no_improv == False:
            # self.input_layer = self.input_layer.cpu()
            self.input_layer = self.input_layer.cuda()
            self.idx = curr_idx
            
            set_actions, action, reward, curr_perf, layer_name, critic_eval = transformer_trainer.validation(self.input_layer)

            print('eval', action, layer_name, critic_eval[0])
            all_prev_actions.append(set_actions)
            all_rewards.append(reward)
            all_perfs.append(curr_perf)
            all_layers.append(layer_name)
            all_critic_eval.append(critic_eval[0].detach().cpu().numpy())

            curr_idx += 1
        df_res = pd.DataFrame()
        df_res['action_0'] = [prev_action[0][0] for prev_action in all_prev_actions]
        df_res['action_1'] = [prev_action[1][0] for prev_action in all_prev_actions]
        df_res['action_2'] = [prev_action[2][0] for prev_action in all_prev_actions]
        df_res['action_3'] = [prev_action[3][0] for prev_action in all_prev_actions]
        df_res['action_4'] = [prev_action[4][0] for prev_action in all_prev_actions]
        df_res['action_5'] = [prev_action[5][0] for prev_action in all_prev_actions]
        df_res['reward'] = all_rewards
        df_res['critic_eval'] = all_critic_eval
        df_res['perf'] = all_perfs
        df_res['layers'] = all_layers
        df_res.to_csv(f"{RESULTS_DIR}/exp{self.exp_number}/final_results/prel_res_{episode}.csv")
        
        self.reset_values()

        return df_res
    
    def interpretability(self, episode, transformer_trainer: TransformerTrainer):
        set_columns = [f"attn_{i}" for i in range(self.max_layers)]
        set_columns.extend(["episode", 'critic_eval', 'reward', 'perf', 'curr_idx'])
        df_res = pd.DataFrame(columns = set_columns)
        df_layer = pd.DataFrame(columns = ["layer", "episode", "curr_idx", "perf"])
        curr_idx = 0
        while(curr_idx <= (self.max_layers - 1) and self.invalid_model == False and self.no_improv == False):
            self.input_layer = self.input_layer.cuda()
            self.idx = curr_idx
            set_actions, action, reward, curr_perf, layer_name, critic_eval = transformer_trainer.validation(self.input_layer)
            example_act = transformer_trainer.critic.encoder.layers[0].attention.scores #the returned [1, 4, 8, 16] (number_heads, seq_length, single_head_dim)
            example_act = torch.mean(F.softmax(example_act, dim = 1), dim = 2)
            attention_map_np = example_act.detach().cpu().numpy()
            res_np = attention_map_np[0]
            
            res_np = np.concatenate((res_np, np.array([episode, critic_eval[0].item(), reward, curr_perf, curr_idx])))
            res_np = res_np.reshape(1, -1)
            ls = pd.DataFrame(res_np, columns=set_columns)
            # print('layer', [layer_name, episode, curr_idx, curr_perf])a
            arr_layer = [layer_name[0], episode, curr_idx, curr_perf]
            print('arr_layer', arr_layer)
            single_layer = pd.DataFrame([arr_layer], columns = df_layer.columns)
            df_layer = pd.concat([df_layer, single_layer], axis = 0)
            df_res = pd.concat([df_res, ls], axis = 0)
            
            print('eval', curr_perf, layer_name, reward)
            curr_idx += 1

        self.reset_values()

        return df_res, df_layer


    def step(self, action, state_encoder = None):
        self.set_actions[self.idx] = torch.from_numpy(np.array(action, dtype=float))
        new_model, layer = self.build_architecture(self.set_actions, self.idx)

        output_reward = 0
        old_state = self.input_layer.clone()
        curr_perf = self.prev_perf
        curr_acc = -1

        if new_model == None:
            output_reward, curr_perf = torch.mul(torch.ones(size = (1, self.performace_pts)), -1).reshape(-1,), torch.mul(torch.ones(size = (1, self.performace_pts)), -1).reshape(-1,)
            self.invalid_model = True
        else:
            lr_r = 1e-4 if self.is_testing else 1e-3

            optimizer = torch.optim.Adam(new_model.parameters(), lr = lr_r)
            perf_arr = np.zeros(shape = (1, self.performace_pts))

            _ = self.train_new_model(new_model, optimizer)
            perf_arr = self.validate_model(new_model)


            del new_model

            output_reward = torch.subtract(torch.tensor(perf_arr), self.prev_perf).reshape(-1,)
            if torch.mean(output_reward) < self.threshold or np.average(perf_arr) < self.eval_threshold:
                self.no_improv = True
                print(f'average: {np.average(perf_arr)}, eval_threshold: {self.eval_threshold}')
            curr_perf = torch.from_numpy(np.array(perf_arr)).float().reshape(-1,)
            self.prev_perf = torch.from_numpy(np.array(perf_arr)).float().reshape(-1,)

        self.prev_actions = torch.from_numpy(np.array(action)).float()

        new_state = self.input_layer
        if state_encoder == None:
            new_state_encoding = self.state_encoder(self.prev_actions, self.prev_perf)
        else:
            new_state_encoding = state_encoder(self.prev_actions, self.prev_perf)
            
        new_state[:, self.idx, :] = new_state_encoding
        self.input_layer = new_state
        next_state = new_state

        done = True if (self.idx == (self.max_layers - 1) or self.invalid_model == True or self.no_improv == True) else False
        self.prev_arch = self.curr_arch
        
        return old_state, self.idx, action, np.array(self.set_actions), output_reward, next_state, curr_perf, curr_acc, self.prev_layer_name, self.prev_arch, layer, done


