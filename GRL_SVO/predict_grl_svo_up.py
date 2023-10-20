import os
import time
import copy
import torch
import argparse
import gym
import cad_order_gym
from models.model_up import A2C_Model
from config import EnvP_config, model_config
import random
from cad_order_gym.envs.mp_penv import SubprocVecEnv


seed = 2022
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0, help='cuda index')
parser.add_argument('--batch_size', type=int, default=40, help='batch size')

parser.add_argument('--dataset', type=str, default='dataset_rand3', help='dataset for predict')
parser.add_argument('--step', type=str, default='0', help='have made n step')

parser.add_argument('--model_dir', type=str, default='train_result/up', help='predict model dir')
parser.add_argument('--best_epoch', type=str, default='100')

parser.add_argument('--information', type=str, default='up_rand3', help='something to mark the predict')


if __name__ == '__main__':
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.cuda) if use_cuda else "cpu")
    batch_size = args.batch_size

    dataset = args.dataset
    step = int(args.step)
    next_step = step + 1

    best_epoch = int(args.best_epoch)
    model_dir = args.model_dir
    
    information = args.information

    predict_list = os.listdir('datasets/' + dataset + f'/projection/feat_{step}')
    
    remain_time_dict = {}
    for file in predict_list:
        index_file = file[:file.rfind('.')]
        with open('datasets/' + dataset + f'/projection/time_{step}/{index_file}.remain') as f:
            time_data = f.readline().strip()
        remain_time_dict[index_file] = float(time_data)

    with open(model_dir + '/information.txt', 'r') as f:
        lines = f.readlines()

    # experiment informations:
    # train_time = 1
    # cuda = 1
    # lr = 2e-05
    # num_convs = 4
    # hidden_channels = 256
    # actor_mlp_layers = 3
    # actor_mlp_list = [256, 512, 128, 1]
    # critic_mlp_layers = 3
    # critic_mlp_list = [256, 512, 128, 1]
    # total_epoch = 80
    # batch_size = 32
    # norm_mode = zscore
    # max_cells = 20000

    train_time = int(lines[1][lines[1].find('=') + 2:-1])
    num_convs = int(lines[4][lines[4].find('=') + 2:-1])
    hidden_channels = int(lines[5][lines[5].find('=') + 2:-1])
    actor_mlp_layers = int(lines[6][lines[6].find('=') + 2:-1])
    
    actor_mlp_list = lines[7][lines[7].find('=') + 2:-1][1:-1].split(', ')
    actor_mlp_list = [int(actor_mlp_list[i]) for i in range(len(actor_mlp_list))]

    critic_mlp_layers = int(lines[8][lines[8].find('=') + 2:-1])
    
    critic_mlp_list = lines[9][lines[9].find('=') + 2:-1][1:-1].split(', ')
    critic_mlp_list = [int(critic_mlp_list[i]) for i in range(len(critic_mlp_list))]

    norm_mode = lines[12][lines[12].find('=') + 2:-1]
    

    nn_config = model_config(num_convs = num_convs,
                            hidden_channels = hidden_channels,
                            actor_mlp_layers = actor_mlp_layers,
                            actor_mlp_list = actor_mlp_list,
                            critic_mlp_layers = critic_mlp_layers,
                            critic_mlp_list = critic_mlp_list)
    a2c = A2C_Model(nn_config)
    params = torch.load(model_dir + f'/checkpoints/a2c_{best_epoch}.pth', map_location=device)
    a2c.load_state_dict(params)
    a2c.to(device)

    if not os.path.exists(f'predict_result/{information}'):
        os.makedirs(f'predict_result/{information}')

    with open(f'predict_result/{information}/action_{next_step}.log', 'w') as f:
        f.write('predict informations:\n')
        f.write(f'best epoch: {best_epoch}\n')
        f.write(f'dataset: {dataset}\n')
        f.write('\n')

    def make_env(config: EnvP_config):
        def _thunk():
            env = gym.make('cad_order_pred_up-v0', config=config)
            return env
        return _thunk

    predict_total_batch = len(predict_list) // batch_size
    predict_residue_num = len(predict_list) - predict_total_batch * batch_size

    auxiliary_config = EnvP_config(dataset=dataset, norm_mode=norm_mode, step=step)
    env_configs = [copy.deepcopy(auxiliary_config) for _ in range(batch_size)]
    envs_list = [make_env(env_configs[i]) for i in range(batch_size)]
    envs = SubprocVecEnv(envs_list)

    if predict_residue_num:
        predict_res_env_configs = [copy.deepcopy(auxiliary_config) for _ in range(predict_residue_num)]
        predict_res_envs_list = [make_env(predict_res_env_configs[i]) for i in range(predict_residue_num)]
        predict_res_envs = SubprocVecEnv(predict_res_envs_list)

    a2c.model.eval()

    informations = []

    for batch in range(predict_total_batch):
        print('predict_batch: ', batch)
        
        for i in range(batch_size):
            index_file = predict_list[batch * batch_size + i][:predict_list[batch * batch_size + i].rfind('.')]
            env_configs[i].change(file=index_file,
                                  remain_time=remain_time_dict[index_file])
            assert envs.re_create(i, env_configs[i])
        
        states = envs.reset()
        a2c.states_ = [copy.deepcopy(state) for state in states]
        a2c.states_to(device)
        a2c.follow()
        dones = [False] * batch_size
        
        temp_dicts = envs.get_index_map_var()

        nn_start_time = time.time()
        with torch.no_grad():
            probs, critics, actions = a2c.get_greedy_actions(dones)
        network_time = time.time() - nn_start_time

        str_actions = [temp_dicts[i][actions[i]] if not dones[i] else actions[i] for i in range(batch_size)]
        
        predict_datas = envs.predict_finish_data(str_actions)
        for predict_data in predict_datas:
            file = predict_data[0]
            action = predict_data[1]
            remain_time = str(float(predict_data[2]) - network_time / batch_size)
            others = predict_data[3]
            informations.append(f'{file}.poly {action} {remain_time} {others}\n')


    if predict_residue_num != 0:
        print('predict_batch: ', predict_total_batch)

        for i in range(predict_residue_num):
            index_file = predict_list[predict_total_batch * batch_size + i][:predict_list[predict_total_batch * batch_size + i].rfind('.')]
            predict_res_env_configs[i].change(file=index_file,
                                              remain_time=remain_time_dict[index_file])
            assert predict_res_envs.re_create(i, predict_res_env_configs[i])

        states = predict_res_envs.reset()
        a2c.states_ = [copy.deepcopy(state) for state in states]
        a2c.states_to(device)
        a2c.follow()                
        dones = [False] * predict_residue_num

        temp_dicts = predict_res_envs.get_index_map_var()

        nn_start_time = time.time()
        with torch.no_grad():
            probs, critics, actions = a2c.get_greedy_actions(dones)
        network_time = time.time() - nn_start_time

        str_actions = [temp_dicts[i][actions[i]] if not dones[i] else actions[i] for i in range(predict_residue_num)]
        
        predict_datas = predict_res_envs.predict_finish_data(str_actions)
        for predict_data in predict_datas:
            file = predict_data[0]
            action = predict_data[1]
            remain_time = str(float(predict_data[2]) - network_time / predict_residue_num)
            others = predict_data[3]
            informations.append(f'{file}.poly {action} {remain_time} {others}\n')

    with open(f'predict_result/{information}/action_{next_step}.log', 'a') as f:
        for info in informations:
            f.write(info)
