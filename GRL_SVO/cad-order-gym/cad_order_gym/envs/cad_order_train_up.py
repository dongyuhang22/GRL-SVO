import copy
import gym
import os
import random
from cad_order_gym.envs.train_processor import Processor
from itertools import permutations
from cad_order_gym.envs.projection_commands import Projection


class CADEnvTUP(gym.Env):
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
        self.pre_vars = None # projection used


    def step(self, action): 
        reward = 0
        # be careful: deepcopy!!!
        if not self.done:
            if len(self.current_vars) == 2:
                self.done = True
                self.all_actions += action
                self.current_vars.pop(self.current_vars.index(action))
                self.all_actions += self.current_vars[0]
                reward = self.cell_dict[self.all_actions] / self.config.max_cells
                return self.state, reward, self.done, {}

            self.pre_vars.remove(action)
            lvar_mark = ','.join(list(self.pre_vars))

            if not os.path.exists(f'datasets/dataset_{self.config.mode}/projection/txt/{self.config.file}_{self.all_actions}{action}.txt'):
                Projection(str(self.instance), action, lvar_mark, self.config, self.all_actions)

            with open(f'datasets/dataset_{self.config.mode}/projection/txt/{self.config.file}_{self.all_actions}{action}.txt') as f:
                str_polys = f.readline().strip()[1:-1].split(', ')
            assert str_polys[0] != 'projs_fac'

            if str_polys[0] != '':
                self.state = copy.deepcopy(self.processor.process_polys(str_polys,
                                                                        norm_mode=self.config.norm_mode,
                                                                        embedding=self.config.have_embedding, 
                                                                        edge=self.config.have_edge,
                                                                        feature_mode=self.config.feature_mode))
                self.instance = copy.deepcopy(self.processor.instance)
                self.current_vars = copy.deepcopy(self.instance.get_vars())
                self.var_map_index = copy.deepcopy(self.processor.name2vari)
                self.index_map_var = {}
                for key in self.var_map_index:
                    self.index_map_var[self.var_map_index[key]] = key
                
                projection_vars = self.pre_vars - set(self.current_vars)
                projection_vars = list(projection_vars)
                random.shuffle(projection_vars)
                projection_vars.insert(0, action)
                self.pre_vars = set(copy.deepcopy(self.current_vars))

                for var in projection_vars:
                    self.all_actions += var

                # print(self.all_actions)
                
                if len(self.current_vars) == 1:
                    self.all_actions += self.current_vars[0]
                    reward = self.cell_dict[self.all_actions] / self.config.max_cells
                    self.done = True
            
            else:
                projection_vars = list(self.pre_vars)
                random.shuffle(projection_vars)
                projection_vars.insert(0, action)

                for var in projection_vars:
                    self.all_actions += var

                reward = self.cell_dict[self.all_actions] / self.config.max_cells
                self.done = True

        return self.state, reward, self.done, {}


    def reset(self):
        self.state = copy.deepcopy(self.processor.process_file(f'datasets/dataset_{self.config.mode}/polys/{self.config.file}.poly',
                                                               norm_mode=self.config.norm_mode, 
                                                               embedding=self.config.have_embedding, 
                                                               edge=self.config.have_edge,
                                                               feature_mode=self.config.feature_mode))
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

        self.pre_vars = set(copy.deepcopy(self.processor.instance.get_vars())) # projection used
        
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
        self.pre_vars = None # projection used

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
