import torch
import gym
from itertools import combinations
from torch_geometric.data import Data
from cad_order_gym.envs.projection_commands import *
import time
import numpy as np


class CADEnvPUP(gym.Env):
    def __init__(self, config):
        self.config = config
        self.state = None
        self.current_vars = None
        self.action = ''
        self.others = ''
        self.index_map_var = None
        self.var_map_index = None        
        self.masks = {
                        'train': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        'Chen': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                        'england': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                        'max': [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
                        'sum': [1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        'prop':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                        'var': [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                        'term': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
                        'poly': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        'degree': [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                        'not_degree': [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1]
                     }
        for i in range(14):
            self.masks['feature_' + str(i + 1)] = [0] * i + [1] + [0] * (13 - i)


    def step(self, action): 
        return None, 0, True, {}


    def reset(self):
        reset_start_time = time.time()
        if self.config.step == 0:
            feature_file = 'datasets/' + self.config.dataset + f'/projection/feat_{self.config.step}/' + self.config.file + '.f'
        else:
            feature_file = 'datasets/' + self.config.dataset + f'/projection/{self.config.model_dir}_feat_{self.config.step}/' + self.config.file + '.f'
        with open(feature_file) as f:
            lines = f.readlines()
            for index in range(len(lines)):
                lines[index] = lines[index].strip()
        state_data = ''.join(lines)

        # assignment
        # [[[[2, 2, 2, 4, 3, 2, 5, 12, 4, 3, 1, 1/3, 2, 2], [2, 2, 2, 3, 4, 3, 5, 15, 6, 5, 1, 5/9, 2, 2], [2, 2, 2, 3, 3, 3, 5, 18, 6, 6, 1, 2/3, 2, 2]], [[x1, x2, x3]]], {x1, x2, x3}]

        state_datas = state_data.split(', {')
        self.current_vars = state_datas[1][:-2].split(', ')
        self.current_vars.sort()
        self.var_map_index = {}
        self.index_map_var = {}
        for var in self.current_vars:
            self.var_map_index[var] = len(self.var_map_index)
            self.index_map_var[len(self.index_map_var)] = var
        self.state = self.get_state(state_datas[0][4:-3])

        self.config.remain_time -= (time.time() - reset_start_time)
        return self.state


    def get_state(self, state_data):
        if ']], [[' in state_data:
            state_data = state_data.split(']], [[')
            features = state_data[0].split('], [')
            every_poly_vars = state_data[1].split('], [')

            for feature_index in range(len(features)):
                if self.config.have_embedding:
                    features[feature_index] = features[feature_index].split(', ')
                    for dim_index in range(14):
                        features[feature_index][dim_index] = eval(features[feature_index][dim_index])
                    features[feature_index] = np.array(features[feature_index])
                    features[feature_index] = (features[feature_index] * self.masks[self.config.feature_mode]).tolist()
                else:
                    features[feature_index] = [1] * 14
            features = self.normalize(torch.tensor(features, dtype=torch.float), self.config.norm_mode)

            for poly_var_index in range(len(every_poly_vars)):
                every_poly_vars[poly_var_index] = every_poly_vars[poly_var_index].split(', ')
            total_edge = set()
            for every_poly_var in every_poly_vars:
                total_edge |= set(combinations(every_poly_var, 2))
            total_edge = list(total_edge)

            edge_index = [[], []]
            if self.config.have_edge:
                for var1, var2 in total_edge:
                    edge_index[0].append(self.var_map_index[var1])
                    edge_index[1].append(self.var_map_index[var2])
                    edge_index[0].append(self.var_map_index[var2])
                    edge_index[1].append(self.var_map_index[var1])
            edge_index = torch.tensor(edge_index, dtype=torch.long)

        else:
            features = state_data[:-4].split('], [')
            for feature_index in range(len(features)):
                if self.config.have_embedding:
                    features[feature_index] = features[feature_index].split(', ')
                    for dim_index in range(14):
                        features[feature_index][dim_index] = eval(features[feature_index][dim_index])
                    features[feature_index] = np.array(features[feature_index])
                    features[feature_index] = (features[feature_index] * self.masks[self.config.feature_mode]).tolist()
                else:
                    features[feature_index] = [1] * 14
            features = self.normalize(torch.tensor(features, dtype=torch.float), self.config.norm_mode)

            edge_index = torch.tensor([[], []], dtype=torch.long)

        return Data(x=features, edge_index=edge_index)


    def normalize(self, tensor, mode):
        if mode == 'zscore':
            if tensor.shape[0] == 1:
                return tensor / (torch.max(torch.abs(tensor), dim=0)[0] + 1e-07)
            else:
                return (tensor - torch.mean(tensor, dim=0)) / (torch.std(tensor, dim=0) + 1e-07)
        else:
            raise ValueError('invalid normalize mode!')


    def predict_finish_data(self, action):
        self.action = action
        self.others = list(set(self.current_vars) - {self.action})
        self.others.sort()
        self.others = ','.join(self.others)        
        return (self.config.file, self.action, self.config.remain_time, self.others)


    def re_create(self, config):
        self.config = config
        self.state = None
        self.current_vars = None
        self.action = ''
        self.others = ''
        self.index_map_var = None
        self.var_map_index = None        

        return True


    def get_index_map_var(self):
        return self.index_map_var


    def close(self):
        return None


    def seed(self, seed=None):
        pass
