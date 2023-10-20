import torch
import gym
from itertools import combinations
from torch_geometric.data import Data
from cad_order_gym.envs.projection_commands import *
import time
import numpy as np


class CADEnvPNUP(gym.Env):
    def __init__(self, config):
        self.config = config
        self.state = None
        self.current_vars = None
        self.all_actions = ''
        self.done = False
        self.var_map_index = None
        self.index_map_var = None
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
        step_time_start = time.time()
        # the index of the new state's emb that need to be updated
        update_embedding_index = []
        # delete emb's index in the old state's emb
        delete = -1
        # the tensor index of the new state's emb in the old state's emb
        tensor_index = []
        
        done_flag = 0
        if not self.done:
            done_flag = 1
            self.all_actions += action

            # note: current_vars are arranged by dictionary order
            delete = self.current_vars.index(action)
            self.current_vars.pop(delete)

            # after pop, current_vars are arranged still by dictionary order
            # get embeddings' index
            tensor_index = [self.var_map_index[self.current_vars[var]] for var in range(len(self.current_vars))]
            # update
            x = self.state.x[tensor_index]

            temp_edge_index = self.state.edge_index.tolist()
            edge_index = [[], []]
            len_temp_edge = len(temp_edge_index[0])
            len_current_vars = len(self.current_vars)
            # up to now, var_map_index and index_to_var are not updated, state will be not changed.

            # new vars map indexes
            self.var_map_index = {}
            for num in range(len_current_vars):
                self.var_map_index[self.current_vars[num]] = num

            # make edge
            for num in range(len_temp_edge):
                if temp_edge_index[0][num] == delete:
                    update_embedding_index.append(self.var_map_index[self.index_map_var[temp_edge_index[1][num]]])

                if temp_edge_index[0][num] != delete and temp_edge_index[1][num] != delete:
                    edge_index[0].append(self.var_map_index[self.index_map_var[temp_edge_index[0][num]]])
                    edge_index[1].append(self.var_map_index[self.index_map_var[temp_edge_index[1][num]]])

            edge_index = torch.tensor(edge_index, dtype=torch.long)

            self.state = Data(x=x, edge_index=edge_index)

            self.index_map_var = {}
            for key in self.var_map_index:
                self.index_map_var[self.var_map_index[key]] = key

            if len_current_vars == 1:
                self.done = True
                self.all_actions += self.current_vars[0]

        if done_flag:
            self.config.remain_time -= (time.time() - step_time_start)

        return self.state, 0, self.done, (update_embedding_index, delete, tensor_index)


    def reset(self):
        reset_start_time = time.time()
        feature_file = 'datasets/' + self.config.dataset + f'/projection/feat_{self.config.step}/' + self.config.file + '.f'
        with open(feature_file) as f:
            lines = f.readlines()
            for index in range(len(lines)):
                lines[index] = lines[index].strip()
        state_data = ''.join(lines)
        
        # assignment
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
        return (self.config.file, self.all_actions, self.config.remain_time)


    def re_create(self, config):
        self.config = config
        self.state = None
        self.current_vars = None
        self.all_actions = ''
        self.done = False
        self.var_map_index = None
        self.index_map_var = None

        return True


    def get_index_map_var(self):
        return self.index_map_var


    def close(self):
        return None


    def seed(self, seed=None):
        pass
