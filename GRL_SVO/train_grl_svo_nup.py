import os
import copy
import torch
import argparse
import gym
import cad_order_gym
import torch.optim as optim
import matplotlib.pyplot as plt
from models.model_nup import A2C_Model_NUP
from config import EnvT_config, model_nup_config
import random
from cad_order_gym.envs.mp_tenv import SubprocVecEnv


seed = 2022
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0, help='cuda index')
parser.add_argument('--lr', type=str, default='2e-05', help='learning rate of a2c')

parser.add_argument('--num_convs', type=int, default=4, help='number of gnn convs')
parser.add_argument('--hidden_channels', type=int, default=256, help='hidden channels of gnn convs')
parser.add_argument('--origin_mlp_layers', type=int, default=3, help='linear num of origin emb_model')
parser.add_argument('--origin_mlp_list', type=list, default=[14, 256, 128, 64], help='channel list of origin emb_model')
parser.add_argument('--neighbor_mlp_layers', type=int, default=3, help='linear num of neighbor emb_model')
parser.add_argument('--neighbor_mlp_list', type=list, default=[128, 512, 256, 64], help='channel list of neighbor emb_model')
parser.add_argument('--actor_mlp_layers', type=int, default=3, help='linear num of actor')
parser.add_argument('--actor_mlp_list', type=list, default=[256, 512, 128, 1], help='channel list of actor')
parser.add_argument('--critic_mlp_layers', type=int, default=3, help='linear num of critic')
parser.add_argument('--critic_mlp_list', type=list, default=[256, 512, 128, 1], help='channel list of critic')

parser.add_argument('--epoch', type=str, default='100', help='total epoch')
parser.add_argument('--batch_size', type=str, default='32', help='batch size')

parser.add_argument('--norm_mode', type=str, default='zscore', help='normalize mode')
parser.add_argument('--max_cells', type=int, default=50000, help='max cells')
parser.add_argument('--train_time', type=int, default=1, help='help to mark train results')


if __name__ == '__main__':
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.cuda) if use_cuda else "cpu")
    lr = float(args.lr)

    num_convs = args.num_convs
    hidden_channels = args.hidden_channels
    actor_mlp_layers = args.actor_mlp_layers
    actor_mlp_list = args.actor_mlp_list
    critic_mlp_layers = args.critic_mlp_layers
    critic_mlp_list = args.critic_mlp_list
    origin_mlp_layers = args.origin_mlp_layers
    origin_mlp_list = args.origin_mlp_list
    neighbor_mlp_layers = args.neighbor_mlp_layers
    neighbor_mlp_list = args.neighbor_mlp_list

    total_epoch = int(args.epoch)
    batch_size = int(args.batch_size)

    norm_mode = args.norm_mode
    max_cells = int(args.max_cells)
    train_time = args.train_time

    train_list = os.listdir(f'datasets/dataset_train/polys')
    valid_list = os.listdir(f'datasets/dataset_valid/polys')

    dir_basic = f'nup_{train_time}'
    
    nn_config = model_nup_config(num_convs = num_convs,
                                 hidden_channels = hidden_channels,
                                 actor_mlp_layers = actor_mlp_layers,
                                 actor_mlp_list = actor_mlp_list,
                                 critic_mlp_layers = critic_mlp_layers,
                                 critic_mlp_list = critic_mlp_list,
                                 origin_mlp_layers = origin_mlp_layers,
                                 origin_mlp_list = origin_mlp_list,
                                 neighbor_mlp_layers = neighbor_mlp_layers,
                                 neighbor_mlp_list = neighbor_mlp_list)
    a2c = A2C_Model_NUP(nn_config)
    a2c.to(device)

    optimizer = optim.Adam(a2c.parameters(), lr)

    cp_dir = 'train_result/'+ dir_basic + '/checkpoints'
    timeout_dir = 'train_result/' + dir_basic + '/timeout'
    time_dir = 'train_result/' + dir_basic + '/times'
    cell_dir = 'train_result/' + dir_basic + '/cells'
    loss_dir = 'train_result/' + dir_basic + '/loss'
    for _ in [cp_dir, timeout_dir, time_dir, cell_dir, loss_dir]:
        if not os.path.exists(_):
            os.makedirs(_)

    with open('train_result/' + dir_basic + '/information.txt', 'w') as f:
        f.write('experiment informations:\n')
        f.write(f'train_time = {train_time}\n')
        f.write(f'cuda = {args.cuda}\n')
        f.write(f'lr = {args.lr}\n')
        f.write(f'num_convs = {num_convs}\n')
        f.write(f'hidden_channels = {hidden_channels}\n')
        f.write(f'actor_mlp_layers = {actor_mlp_layers}\n')
        f.write(f'actor_mlp_list = {actor_mlp_list}\n')
        f.write(f'critic_mlp_layers = {critic_mlp_layers}\n')
        f.write(f'critic_mlp_list = {critic_mlp_list}\n')
        f.write(f'origin_mlp_layers = {origin_mlp_layers}\n')
        f.write(f'origin_mlp_list = {origin_mlp_list}\n')
        f.write(f'neighbor_mlp_layers = {neighbor_mlp_layers}\n')
        f.write(f'neighbor_mlp_list = {neighbor_mlp_list}\n')
        f.write(f'total_epoch = {total_epoch}\n')
        f.write(f'batch_size = {batch_size}\n')
        f.write(f'norm_mode = {norm_mode}\n')
        f.write(f'max_cells = {max_cells}\n')

    def make_env(config: EnvT_config):
        def _thunk():
            env = gym.make('cad_order_train_nup-v0', config=config)
            return env
        return _thunk

    train_timeout = []
    train_times = []
    train_cells = []
    train_losses = []
    valid_timeout = []
    valid_times = []
    valid_cells = []

    train_batch_num = len(train_list) // batch_size
    valid_batch_num = len(valid_list) // batch_size

    train_residue_num = len(train_list) - train_batch_num * batch_size
    valid_residue_num = len(valid_list) - valid_batch_num * batch_size

    auxiliary_config = EnvT_config(norm_mode=norm_mode, max_cells=max_cells)
    env_configs = [copy.deepcopy(auxiliary_config) for _ in range(batch_size)]
    envs_list = [make_env(env_configs[i]) for i in range(batch_size)]
    envs = SubprocVecEnv(envs_list)

    if train_residue_num:
        train_res_env_configs = [copy.deepcopy(auxiliary_config) for _ in range(train_residue_num)]
        train_res_envs_list = [make_env(train_res_env_configs[i]) for i in range(train_residue_num)]
        train_res_envs = SubprocVecEnv(train_res_envs_list)

    if valid_residue_num:
        valid_res_env_configs = [copy.deepcopy(auxiliary_config) for _ in range(valid_residue_num)]
        valid_res_envs_list = [make_env(valid_res_env_configs[i]) for i in range(valid_residue_num)]
        valid_res_envs = SubprocVecEnv(valid_res_envs_list)

    for epoch in range(1, total_epoch + 1):
        random.shuffle(train_list)
        float_loss = 0
        for batch in range(train_batch_num):
            for i in range(batch_size):
                env_configs[i].change(file=train_list[batch * batch_size + i][:train_list[batch * batch_size + i].find('.')], 
                                      mode='train')
                assert envs.re_create(i, env_configs[i])

            a2c.model.train()
            total_rewards = []
            total_probs = []
            total_critics = []
            dones = [False] * batch_size
            
            states = envs.reset()
            a2c.states_ = [copy.deepcopy(state) for state in states]
            a2c.states_to(device)
            a2c.update_embedding()
            a2c.follow()

            while True:
                print('EPOCH: ', epoch, 'batch: ', batch, ' step')
                temp_dicts = envs.get_index_map_var()

                probs, critics, actions = a2c.get_random_actions(dones)
                str_actions = [temp_dicts[i][actions[i]] if not dones[i] else actions[i] for i in range(batch_size)]
                
                states, rewards, dones, infos = envs.step(str_actions)                
                a2c.states_ = [copy.deepcopy(state) for state in states]
                a2c.states_to(device)
                a2c.update_embedding(infos)

                total_probs.append(probs)
                total_critics.append(critics)
                total_rewards.append(rewards)

                if dones == [True] * batch_size:
                    break
                
                a2c.follow()

            advantage = []
            total_ps = [1] * batch_size
            len_probs = len(total_probs)
            actor_loss = 0
            critic_loss = 0

            for i in range(batch_size):
                i_reward = []
                for j in range(len_probs):
                    total_ps[i] = total_ps[i] * total_probs[j][i]
                    i_reward.append(total_rewards[j][i])
                
                total_ps[i] = torch.log(total_ps[i])

                advantage.append(torch.tensor(sum(i_reward)) - total_critics[0][i])
                actor_loss = actor_loss + advantage[-1].detach() * total_ps[i][0] / batch_size
                critic_loss = critic_loss + advantage[-1].pow(2) / batch_size

            loss = actor_loss + critic_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            float_loss += float(loss)

        if train_residue_num != 0:
            print('EPOCH:', epoch, 'batch:', train_batch_num)

            for i in range(train_residue_num):
                train_res_env_configs[i].change(file=train_list[train_batch_num * batch_size + i][:train_list[train_batch_num * batch_size + i].find('.')], 
                                                mode='train')
                assert train_res_envs.re_create(i, train_res_env_configs[i])

            a2c.model.train()
            total_rewards = []
            total_probs = []
            total_critics = []
            dones = [False] * train_residue_num
            
            states = train_res_envs.reset()
            a2c.states_ = [copy.deepcopy(state) for state in states]
            a2c.states_to(device)
            a2c.update_embedding()
            a2c.follow()
            
            while True:

                temp_dicts = train_res_envs.get_index_map_var()
                probs, critics, actions = a2c.get_random_actions(dones)

                str_actions = [temp_dicts[i][actions[i]] if not dones[i] else actions[i] for i in range(train_residue_num)]
                
                states, rewards, dones, infos = train_res_envs.step(str_actions)
                a2c.states_ = [copy.deepcopy(state) for state in states]
                a2c.states_to(device)
                a2c.update_embedding(infos)
                
                total_probs.append(probs)
                total_critics.append(critics)
                total_rewards.append(rewards)

                if dones == [True] * train_residue_num:
                    break

                a2c.follow()

            advantage = []
            total_ps = [1] * train_residue_num
            len_probs = len(total_probs)
            actor_loss = 0
            critic_loss = 0

            for i in range(train_residue_num):
                i_reward = []
                for j in range(len_probs):
                    total_ps[i] = total_ps[i] * total_probs[j][i]
                    i_reward.append(total_rewards[j][i])  
                
                total_ps[i] = torch.log(total_ps[i])
                advantage.append(torch.tensor(sum(i_reward)) - total_critics[0][i])
                actor_loss = actor_loss + advantage[-1].detach() * total_ps[i][0] / train_residue_num
                critic_loss = critic_loss + advantage[-1].pow(2) / train_residue_num

            loss = actor_loss + critic_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            float_loss += float(loss)

        if epoch % 2 == 0:
            torch.save(a2c.state_dict(), f'{cp_dir}/a2c_{str(epoch)}.pth')
            train_losses.append(float_loss / (train_batch_num + 1))
            
            a2c.model.eval()
            local_train_average_time = 0
            local_train_average_cell = 0
            local_train_timeout = 0            
            local_valid_average_time = 0
            local_valid_average_cell = 0
            local_valid_timeout = 0
            
            for batch in range(train_batch_num):
                print('train_batch: ', batch)
                for i in range(batch_size):
                    env_configs[i].change(file=train_list[batch * batch_size + i][:train_list[batch * batch_size + i].find('.')], 
                                          mode='train')
                    assert envs.re_create(i, env_configs[i])
                
                states = envs.reset()
                a2c.states_ = [copy.deepcopy(state) for state in states]
                a2c.states_to(device)
                a2c.update_embedding()
                a2c.follow()
                dones = [False] * batch_size
                
                while True:
                    temp_dicts = envs.get_index_map_var()
                    probs, critics, actions = a2c.get_greedy_actions(dones)

                    str_actions = [temp_dicts[i][actions[i]] if not dones[i] else actions[i] for i in range(batch_size)]
                    states, rewards, dones, infos = envs.step(str_actions)

                    a2c.states_ = [copy.deepcopy(state) for state in states]
                    a2c.states_to(device)
                    a2c.update_embedding(infos)
                    
                    if dones == [True] * batch_size:
                        temp_cells = envs.get_final_cells()
                        for temp_cell in temp_cells:
                            if temp_cell == max_cells:
                                local_train_timeout += 1
                            local_train_average_cell += temp_cell
                        temp_times = envs.get_final_time()
                        for temp_time in temp_times:
                            local_train_average_time += temp_time                        
                        break

                    a2c.follow()

            if train_residue_num != 0:
                print('train_batch: ', train_batch_num)
                for i in range(train_residue_num):
                    train_res_env_configs[i].change(file=train_list[train_batch_num * batch_size + i][:train_list[train_batch_num * batch_size + i].find('.')], 
                                                    mode='train')
                    assert train_res_envs.re_create(i, train_res_env_configs[i])
                
                states = train_res_envs.reset()
                a2c.states_ = [copy.deepcopy(state) for state in states]
                a2c.states_to(device)
                a2c.update_embedding()
                a2c.follow()                
                
                dones = [False] * train_residue_num
                while True:
                    temp_dicts = train_res_envs.get_index_map_var()
                    probs, critics, actions = a2c.get_greedy_actions(dones)

                    str_actions = [temp_dicts[i][actions[i]] if not dones[i] else actions[i] for i in range(train_residue_num)]
                    
                    states, rewards, dones, infos = train_res_envs.step(str_actions)
                    a2c.states_ = [copy.deepcopy(state) for state in states]
                    a2c.states_to(device)
                    a2c.update_embedding(infos)                    
                    
                    if dones == [True] * train_residue_num:
                        temp_cells = train_res_envs.get_final_cells()
                        for temp_cell in temp_cells:
                            if temp_cell == max_cells:
                                local_train_timeout += 1
                            local_train_average_cell += temp_cell
                        temp_times = train_res_envs.get_final_time()
                        for temp_time in temp_times:
                            local_train_average_time += temp_time
                        break

                    a2c.follow()

            for batch in range(valid_batch_num):
                print('valid_batch: ', batch)
                for i in range(batch_size):
                    env_configs[i].change(file=valid_list[batch * batch_size + i][:valid_list[batch * batch_size + i].find('.')], 
                                          mode='valid')
                    assert envs.re_create(i, env_configs[i])
                
                states = envs.reset()
                a2c.states_ = [copy.deepcopy(state) for state in states]
                a2c.states_to(device)
                a2c.update_embedding()
                a2c.follow()
                dones = [False] * batch_size
                
                while True:
                    temp_dicts = envs.get_index_map_var()
                    probs, critics, actions = a2c.get_greedy_actions(dones)

                    str_actions = [temp_dicts[i][actions[i]] if not dones[i] else actions[i] for i in range(batch_size)]
                    
                    states, rewards, dones, infos = envs.step(str_actions)
                    a2c.states_ = [copy.deepcopy(state) for state in states]
                    a2c.states_to(device)
                    a2c.update_embedding(infos)
                    
                    if dones == [True] * batch_size:
                        temp_cells = envs.get_final_cells()
                        for temp_cell in temp_cells:
                            if temp_cell == max_cells:
                                local_valid_timeout += 1
                            local_valid_average_cell += temp_cell
                        temp_times = envs.get_final_time()
                        for temp_time in temp_times:
                            local_valid_average_time += temp_time
                        break

                    a2c.follow()

            if valid_residue_num != 0:
                print('valid_batch: ', valid_batch_num)
                for i in range(valid_residue_num):
                    valid_res_env_configs[i].change(file=valid_list[valid_batch_num * batch_size + i][:valid_list[valid_batch_num * batch_size + i].find('.')], 
                                                    mode='valid')
                    assert valid_res_envs.re_create(i, valid_res_env_configs[i])
                
                states = valid_res_envs.reset()
                a2c.states_ = [copy.deepcopy(state) for state in states]
                a2c.states_to(device)
                a2c.update_embedding()
                a2c.follow()                
                
                dones = [False] * valid_residue_num
                while True:
                    temp_dicts = valid_res_envs.get_index_map_var()
                    probs, critics, actions = a2c.get_greedy_actions(dones)

                    str_actions = [temp_dicts[i][actions[i]] if not dones[i] else actions[i] for i in range(valid_residue_num)]
                    
                    states, rewards, dones, infos = valid_res_envs.step(str_actions)                    
                    a2c.states_ = [copy.deepcopy(state) for state in states]
                    a2c.states_to(device)
                    a2c.update_embedding(infos)                    
                    
                    if dones == [True] * valid_residue_num:
                        temp_cells = valid_res_envs.get_final_cells()
                        for temp_cell in temp_cells:
                            if temp_cell == max_cells:
                                local_valid_timeout += 1
                            local_valid_average_cell += temp_cell
                        temp_times = valid_res_envs.get_final_time()
                        for temp_time in temp_times:
                            local_valid_average_time += temp_time
                        break

                    a2c.follow()


            train_times.append(local_train_average_time / len(train_list))
            train_timeout.append(local_train_timeout)
            train_cells.append(local_train_average_cell / len(train_list))
            valid_times.append(local_valid_average_time / len(valid_list))
            valid_timeout.append(local_valid_timeout)
            valid_cells.append(local_valid_average_cell / len(valid_list))            

            with open('train_result/' + dir_basic + '/result.txt', 'w') as f:
                f.write('train_timeout:\n')
                f.write(f'{train_timeout}\n')
                f.write('train_times:\n')
                f.write(f'{train_times}\n')
                f.write('train_cells:\n')
                f.write(f'{train_cells}\n')
                f.write('train_losses:\n')
                f.write(f'{train_losses}\n')
                f.write('valid_timeout:\n')
                f.write(f'{valid_timeout}\n')
                f.write('valid_times:\n')
                f.write(f'{valid_times}\n')
                f.write('valid_cells:\n')
                f.write(f'{valid_cells}\n')                

            print("train_timeout: ", local_train_timeout)
            print("valid timeout: ", local_valid_timeout)

            x = [i for i in range(len(train_timeout))]

            plt.figure()
            plt.plot(x, valid_times, '.-', color='deepskyblue')
            plt.title('AVERAGE_TIME')
            plt.xlabel('EPOCH')
            plt.ylabel('AVERAGE_TIME(s)')
            plt.savefig(f'{time_dir}/valid.jpg')
            plt.close()

            plt.figure()
            plt.plot(x, valid_cells, '.-', color='deepskyblue')
            plt.title('AVERAGE_CELL')
            plt.xlabel('EPOCH')
            plt.ylabel('AVERAGE_CELL')
            plt.savefig(f'{cell_dir}/valid.jpg')
            plt.close()            

            plt.figure()
            plt.plot(x, valid_timeout, '.-', color='deepskyblue')
            plt.title('TIMEOUT')
            plt.xlabel('EPOCH')
            plt.ylabel('TIMEOUT')
            plt.savefig(f'{timeout_dir}/valid.jpg')
            plt.close()

            plt.figure()
            plt.plot(x, train_times, '.-', color='black')
            plt.title('AVERAGE_TIME')
            plt.xlabel('EPOCH')
            plt.ylabel('AVERAGE_TIME(s)')
            plt.savefig(f'{time_dir}/train.jpg')
            plt.close()

            plt.figure()
            plt.plot(x, train_cells, '.-', color='black')
            plt.title('AVERAGE_CELL')
            plt.xlabel('EPOCH')
            plt.ylabel('AVERAGE_CELL')
            plt.savefig(f'{cell_dir}/train.jpg')
            plt.close()            

            plt.figure()
            plt.plot(x, train_timeout, '.-', color='black')
            plt.title('TIMEOUT')
            plt.xlabel('EPOCH')
            plt.ylabel('TIMEOUT')
            plt.savefig(f'{timeout_dir}/train.jpg')
            plt.close()

            plt.figure()
            plt.plot(x, train_losses, '.-', color='black')
            plt.title('LOSS')
            plt.xlabel('EPOCH')
            plt.ylabel('LOSS')
            plt.savefig(f'{loss_dir}/train.jpg')
            plt.close()            
