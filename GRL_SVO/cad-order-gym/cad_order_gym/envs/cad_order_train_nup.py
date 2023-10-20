import copy
import torch
import gym
from cad_order_gym.envs.train_processor import Processor
from itertools import permutations
from torch_geometric.data import Data


class CADEnvTNUP(gym.Env):
    def __init__(self, config):
        self.config = config
        self.processor = Processor()
        self.state = None
        self.instance = None
        self.origin_instance = None
        self.current_vars = None
        self.all_actions = ''
        self.done = False

        self.cell_dict = None
        self.time_dict = None

        self.var_map_index = None
        self.index_map_var = None


    def step(self, action): 
        reward = 0
        # the index of the new state's emb that need to be updated
        update_embedding_index = []
        # delete emb's index in the old state's emb
        delete = -1
        # the tensor index of the new state's emb in the old state's emb
        tensor_index = []
        
        if not self.done:
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
                reward = self.cell_dict[self.all_actions] / self.config.max_cells

        return self.state, reward, self.done, (update_embedding_index, delete, tensor_index)


    def reset(self):
        self.state = copy.deepcopy(self.processor.process_file(f'datasets/dataset_{self.config.mode}/polys/{self.config.file}.poly', 
                                                               norm_mode=self.config.norm_mode,
                                                               embedding=self.config.have_embedding, 
                                                               edge=self.config.have_edge,
                                                               feature_mode=self.config.feature_mode
                                                               ))
        self.instance = copy.deepcopy(self.processor.instance)
        self.origin_instance = copy.deepcopy(self.processor.instance)
        self.current_vars = copy.deepcopy(self.instance.get_vars())

        assert len(self.current_vars) > 1

        self.all_actions = ''
        self.done = False
        
        self.var_map_index = copy.deepcopy(self.processor.name2vari)

        self.index_map_var = {}
        for key in self.var_map_index:
            self.index_map_var[self.var_map_index[key]] = key

        return self.state


    def re_create(self, config):
        self.config = config
        self.processor = Processor()
        self.state = None
        self.instance = None
        self.origin_instance = None
        self.current_vars = None
        self.all_actions = ''
        self.done = False

        self.cell_dict = {}
        self.time_dict = {}
        order_list = list(permutations([f'x{i}' for i in range(1, self.config.var + 1)]))
        for order in order_list:
            order = ''.join(order)
            with open(f'datasets/dataset_{self.config.mode}/cells/{self.config.file}/{self.config.file}_{order}.cad') as f:
                cell_data = f.readline().strip()
            if cell_data == '-1' or cell_data == 'p' or cell_data == 'error':
                self.cell_dict[order] = self.config.max_cells
                self.time_dict[order] = 900
            else:
                self.cell_dict[order] = int(cell_data)
                with open(f'datasets/dataset_{self.config.mode}/times/{self.config.file}/{self.config.file}_{order}.time') as f:
                    time_data = f.readline().strip()
                self.time_dict[order] = float(time_data)

        self.var_map_index = None
        self.index_map_var = None

        return True


    def get_index_map_var(self):
        return self.index_map_var


    def get_var_map_index(self):
        return self.var_map_index


    def get_file(self):
        return self.config.file


    def get_all_actions(self):
        return self.all_actions


    def get_final_time(self):
        return self.time_dict[self.all_actions]


    def get_final_cells(self):
        return self.cell_dict[self.all_actions]        


    def close(self):
        return None


    def seed(self, seed=None):
        pass
