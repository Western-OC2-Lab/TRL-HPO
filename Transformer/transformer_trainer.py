from Transformer.transformer_class import *
from replay_memory import *
from Transformer.transformer_utils import Utils
from functions import *
from helper_classes import *
import time
from joblib import Parallel, cpu_count, delayed
import multiprocessing
# from fvcore.nn import FlopCountAnalysis
from calflops import calculate_flops

class TransformerTrainer:


    def __init__(self, max_layers, embed_dim, num_layers, 
    expansion_factor, n_heads, action_space, size_buffer, env, target_episode, state_encoder, training_loader, testing_loader, saving_dir):
        self.size_buffer = size_buffer
        self.max_layers = max_layers
        self.training_epochs = 5
        self.actor = TransformerActor(seq_len = max_layers, embed_dim = embed_dim, 
        num_layers = num_layers, expansion_factor = expansion_factor, n_heads = n_heads, action_space=action_space)

        self.target_actor = TransformerActor(seq_len = max_layers, embed_dim = embed_dim, 
        num_layers = num_layers, expansion_factor = expansion_factor, n_heads = n_heads, action_space=action_space)

        self.critic = TransformerCritic(seq_len = max_layers, embed_dim = embed_dim, 
        num_layers = num_layers, expansion_factor = expansion_factor, n_heads = n_heads, action_space=action_space)

        self.target_critic = TransformerCritic(seq_len = max_layers, embed_dim = embed_dim, 
        num_layers = num_layers, expansion_factor = expansion_factor, n_heads = n_heads, action_space=action_space)

        self.saving_dir = saving_dir
        self.rm = ReplayMemory(size_buffer)
        self.noise = OrnsteinUhlenbeckActionNoise(4)
        self.lr_c = 1e-4
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = self.lr_c / 10)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = self.lr_c)
        self.training_loader = training_loader
        self.testing_loader = testing_loader
        self.started_training = False
        self.batch_size = 16
        self.losses = {
            'actor_loss': [],
            'critic_loss': []
        }

        self.set_rewards = []
        self.normalization_vars = {
            'min': None,
            'max': None
        }
        self.purge_value = 1000
        self.curr_values = 0
        self.env = env
        self.utils = Utils()
        self.target_episode = target_episode
        self.state_encoder = state_encoder

        self.utils.hard_update(self.target_actor, self.actor)
        self.utils.hard_update(self.target_critic, self.critic)


    def get_ready(self):
        self.actor.train()
        self.critic.train()
        self.target_actor.train()
        self.target_critic.train()
        self.actor.cuda()
        self.critic.cuda()
        self.target_actor.cuda()
        self.target_critic.cuda()

    def get_ready_training(self):
        self.actor.train()
        self.critic.train()
        self.target_actor.train()
        self.target_critic.train()

    def get_exploitation(self, state):
        action = self.actor(state)

        action = action.cpu()
        return action.data.numpy()[0]


    def get_exploration_action(self, state):
        action = self.target_actor(state)
        action = action.cpu()
        noise_sample = self.noise.sample()
        new_action = action.data.numpy()[0] + noise_sample
        new_action = [np.clip(new_action[0], 0, 1), np.clip(new_action[1], 0, 1), np.clip(new_action[2], 0, 1), np.clip(new_action[3], 0, 1)]

        return new_action
    
    
    def new_model_definition(self, source_model, target_model):
        if source_model != None:
            nb_all_layers = len(source_model.all_layers)
            name_layer = "all_layers.{}.{}"
            set_all_eligible_layers = []
            for i in range(nb_all_layers):
                if hasattr(source_model.all_layers[i], "weight"):
                    target_model.all_layers[i].weight.data = source_model.all_layers[i].weight.data
                    set_all_eligible_layers.append(name_layer.format(i, "weight"))
                
                if hasattr(source_model.all_layers[i], "bias"):
                    classname = source_model.all_layers[i].__class__.__name__
                    if classname.find("Linear") != -1: 
                        target_model.all_layers[i].bias = source_model.all_layers[i].bias
                    else:
                        target_model.all_layers[i].bias.data = source_model.all_layers[i].bias.data

                    set_all_eligible_layers.append(name_layer.format(i, "bias"))

            for name, param in target_model.named_parameters():
                if name in set_all_eligible_layers:
                    param.requires_grad = False

        return target_model

    def train_new_model(self, model, optimizer):
        perf_arr = []
        # model = self.new_model_definition(curr_arch, model)
        model.cuda()
        model.train()
        start_time = time.time()
        for epoch in range(self.training_epochs):
            local_perf = []
            for idx, (img, label) in enumerate(self.training_loader):
                optimizer.zero_grad()
                img = img.cuda()
                output = model(img)
                criterion = nn.CrossEntropyLoss()
                loss_value = criterion(output.cpu().float(), label)
                loss_value.backward()
                optimizer.step()
            # if idx % 2175 == 0 and idx != 0:
                accuracy_batch = calculate_accuracy(output.cpu(), label)
                local_perf.append(accuracy_batch)
            print(f'Epoch Accuracy: {np.mean(local_perf)}')
            # perf_arr.append(accuracy_batch)
        
        end_time = time.time()
        total_train_time = round((end_time - start_time)/60, 2)
        # print(f'Transformer training, model: {model}, training_time: {total_train_time}, res: {np.average(perf_arr)}')
        # perf_arr = np.expand_dims(perf_arr[-32:], axis=0)
    
        # print('train_models', perf_arr)
        return perf_arr
    
    def train_actions(self, set_actions, curr_action, set_idx, set_terminals, set_prev_layers, set_prev_archs):
        set_perfs = []
        with torch.no_grad():
            detached_curr_action = curr_action.clone()
        for i_a, action in enumerate(set_actions):
            a_new_set = action
            max_idx = int(set_idx[i_a]) + 1
            prev_arch = set_prev_archs[i_a]
            if max_idx > self.max_layers:
                set_perfs.extend(np.zeros(shape = (1, 32)))
            else:
                curr_idx = max_idx
                a_new_set[curr_idx, :, :] = detached_curr_action[i_a]
                new_model, _ = self.env.build_architecture(a_new_set, curr_idx)
                if new_model == None:
                    set_perfs.extend(np.zeros(shape = (1, 32)))
                else:
                    optimizer = torch.optim.Adam(params=new_model.parameters(), lr = 1e-4)
                    arch_exists, n_perf = self.env.registry.compare_archs(new_model)
                    if arch_exists:
                        perf_action = n_perf
                    else:
                        _ = self.train_new_model(new_model, optimizer, prev_arch)
                    perf_action = self.env.validate_model(new_model)
                    set_perfs.extend(np.expand_dims(np.array(perf_action), axis = 0))
        
        return np.array(set_perfs)
    
    def train_single_action(self, i_a, action, set_idx, detached_curr_action, set_prev_archs):
        a_new_set = action
        max_idx = int(set_idx[i_a]) + 1
        perf_action = np.zeros(shape = (32,))
        if max_idx < self.max_layers:
            curr_idx = max_idx
            # curr_arch = set_prev_archs[i_a]
            curr_arch = []
            a_new_set[curr_idx, :, :] = detached_curr_action[i_a]
            new_model, _ = self.env.build_architecture(a_new_set, curr_idx)
            if new_model != None:
                optimizer = torch.optim.Adam(params=new_model.parameters(), lr = 1e-4)
                _ = self.train_new_model(new_model, optimizer)
                perf_action = self.env.validate_model(new_model)
                    # self.env.registry.add(all_layer_names, perf_action)

            del new_model

        return perf_action
    
    def parallelized_train_actions(self, set_actions, curr_action, set_idx, set_prev_archs):
        with torch.no_grad():
            detached_curr_action = curr_action.clone()
        set_perfs = Parallel(n_jobs= int(3), verbose = 0)(delayed(self.train_single_action)(i_a, action, set_idx, detached_curr_action, set_prev_archs) for i_a, action in enumerate(set_actions))
        
        return np.array(set_perfs)
    
    def optimize(self):
        
        if self.rm.len < (self.size_buffer):
            return

        self.started_training = True
        self.state_encoder.eval()
        state, idx, action, set_actions, reward, next_state, curr_perf, curr_acc, prev_layers, set_prev_archs, done = self.rm.sample(self.batch_size)

        state = torch.from_numpy(state).cuda()
        next_state = torch.from_numpy(next_state).cuda()
        set_actions = torch.from_numpy(np.array(set_actions)).cuda()
        action = torch.from_numpy(action).cuda()
        reward = [np.average(r) for r in reward]
        reward = torch.from_numpy(np.expand_dims(reward, axis = 1))
        reward = reward.cuda()
        done = np.expand_dims(done, axis = 1)
        terminal = torch.from_numpy(done).cuda()

        # ------- optimize critic ----- #
        a_pred = self.target_actor(next_state)
        a_pred = a_pred.detach()
        pred_perf = self.parallelized_train_actions(set_actions, a_pred, idx, set_prev_archs)


        pred_perf = torch.from_numpy(pred_perf)

        new_set_states = torch.Tensor()
        for idx_s, single_state in enumerate(next_state):
            new_state = single_state.clone()
            if terminal[idx_s] == False:
                next_indx = int(idx[idx_s] + 1)
                new_state[next_indx, :] = self.state_encoder(a_pred[idx_s].cpu().float(), pred_perf[idx_s].cpu().float())
            new_state = new_state[None, :]
            new_set_states = torch.cat((new_set_states, new_state.cpu()), dim = 0)
        new_set_states = new_set_states.cuda()
        target_values = torch.add(reward, torch.mul(~terminal, self.target_critic(new_set_states)))
        val_expected = self.critic(next_state)

        criterion = nn.MSELoss()
        loss_critic = criterion(target_values, val_expected)
        self.critic_optimizer.zero_grad()

        loss_critic.backward()

        self.critic_optimizer.step()
        

        # ----- optimize actor ----- #
        pred_a1 = self.actor(next_state)

        # pred_perf = self.train_actions(set_actions, pred_a1, idx, terminal, prev_layers, set_prev_archs)
        pred_perf = self.parallelized_train_actions(set_actions, a_pred, idx, set_prev_archs)

        pred_perf = torch.from_numpy(pred_perf)
        new_set_states = torch.Tensor()
        for idx_s, single_state in enumerate(next_state):
            new_state = single_state.clone()
            if terminal[idx_s] == False:
                next_indx = int(idx[idx_s]+1)
                new_state[next_indx, :] = self.state_encoder(pred_a1[idx_s].cpu().float(), pred_perf[idx_s].cpu().float())
            new_state = new_state[None, :]
            new_set_states = torch.cat((new_set_states, new_state.cpu()), dim = 0) 
        new_set_states = new_set_states.cuda()

        loss_fn = CustomLoss(self.actor, self.critic)
        loss_actor = loss_fn(new_set_states)
        self.actor_optimizer.zero_grad()


        loss_actor.backward()

        self.actor_optimizer.step()

        self.losses['actor_loss'].append(loss_actor.item())
        self.losses['critic_loss'].append(loss_critic.item())
        
        # TAU = 0.001
        TAU = 0.01

        self.utils.soft_update(self.target_actor, self.actor, TAU)
        self.utils.soft_update(self.target_critic, self.critic, TAU)


    def eval(self):
        self.actor.eval()
        self.target_actor.eval()
        self.critic.eval()
        self.target_critic.eval()

        self.actor.cuda()
        self.target_actor.cuda()
        self.critic.cuda()
        self.target_critic.cuda()


    def save_models(self, episode_count):
        
        torch.save(self.target_actor.state_dict(), f'{self.saving_dir}/EP-{episode_count}_target_actor.pt')
        torch.save(self.actor.state_dict(), f'{self.saving_dir}/EP-{episode_count}_actor.pt')

        torch.save(self.target_critic.state_dict(), f'{self.saving_dir}/EP-{episode_count}_target_critic.pt')
        torch.save(self.critic.state_dict(), f'{self.saving_dir}/EP-{episode_count}_critic.pt')
        torch.save(self.state_encoder.state_dict(), f'{self.saving_dir}/EP-{episode_count}_state_encoder.pt')

        print("Models saved successfully")
        
    def load_models(self, model_name):

        self.actor.load_state_dict(torch.load(f'{model_name}_actor.pt'))
        self.critic.load_state_dict(torch.load(f'{model_name}_critic.pt'))

        self.utils.hard_update(self.target_actor, self.actor)
        self.utils.hard_update(self.target_critic, self.critic)


    def validation(self, state):
        self.eval()
        action = self.get_exploitation(state)
        state, idx, action, set_actions, reward, next_state, curr_perf, curr_acc, prev_layers, set_prev_archs, layer, done = self.env.step(action, self.state_encoder)
        
        print(f'idx: {idx}, reward: {np.mean(np.array(reward))}, new_layer: {layer}')
        critic_eval = self.critic(next_state)

        return set_actions, np.array(action), np.array(np.mean(np.array(reward))), np.array(np.mean(curr_perf.numpy())), layer, critic_eval
    
    def normalize_rewards(self, reward):

        r_reward = None
        if self.curr_values == 0:
            self.normalization_vars['max'] = reward[-1]
            self.normalization_vars['min'] = reward[-1]
            r_reward = reward
        elif self.curr_values != 0:
            if reward[-1] < self.normalization_vars['min']:
                self.normalization_vars['min'] = reward[-1]
            if reward[-1] > self.normalization_vars['max']:
                self.normalization_vars['max'] = reward[-1]
            if self.curr_values % self.purge_value == 0:
                self.set_rewards = []
        
            r_reward = (reward - self.normalization_vars['min']) / (self.normalization_vars['max'] - self.normalization_vars['min'])

        print(f'curr_value: {self.curr_values}, reward: {reward}, normalization_vars: {self.normalization_vars}, r_reward: {r_reward}')

        return r_reward



    def train(self, state, episode):
        self.get_ready()
        action = None
        if episode < self.target_episode:
            action = self.get_exploration_action(state)
        else:
            action = self.get_exploitation(state)

        state, idx, action, set_actions, reward, next_state, curr_perf, curr_acc, prev_layer, prev_arch, layer, done = self.env.step(action)
        self.curr_values += 1

        return state, idx, action, set_actions, reward, next_state, curr_perf, curr_acc, prev_layer, prev_arch, layer, done
        
