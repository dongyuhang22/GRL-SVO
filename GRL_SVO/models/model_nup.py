import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import Linear, global_mean_pool, SAGEConv, GraphConv, GATConv, GATv2Conv
from torch_geometric.loader import DataLoader

conv_dict = {
    'SAGE': SAGEConv,
    'GRAPH': GraphConv,
    'GAT': GATConv,
    'GATv2': GATv2Conv
}

class MLP(torch.nn.Module):
    def __init__(self, num_layers, channels_list):
        # parameter example: (3, [14, 256, 128, 64])
        # this will make a MLP with 3 layers: (14, 256), (256, 128), (128, 64)
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(Linear(channels_list[i], channels_list[i + 1]))

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = F.relu(self.convs[i](x))
        return self.convs[-1](x)


class Model_NUP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        # GNN Convs
        self.convs = torch.nn.ModuleList()
        for _ in range(config.num_convs):
            self.convs.append(conv_dict[config.conv](-1, config.hidden_channels))

        # actor & critic
        self.actor = MLP(config.actor_mlp_layers, config.actor_mlp_list)
        self.critic = MLP(config.critic_mlp_layers, config.critic_mlp_list)

        # mlp for origin embeddings
        self.origin = MLP(config.origin_mlp_layers, config.origin_mlp_list)

        # mlp for neighbor embeddings
        self.neighbor = MLP(config.neighbor_mlp_layers, config.neighbor_mlp_list)

    # mode: 'a2c', 'origin', 'neighbor'
    def forward(self, mode, x, edge_index=None, batch=None):
        if mode == 'a2c':
            assert edge_index != None and batch != None
            
            for num in range(len(self.convs)):
                if num != len(self.convs) - 1:
                    x = F.relu(self.convs[num](x, edge_index))
                else:
                    x = self.convs[num](x, edge_index)
            
            actor = self.actor(x)
            critic = self.critic(global_mean_pool(x, batch=batch))

            return actor, critic

        elif mode == 'origin':
            return self.origin(x)

        elif mode == 'neighbor':
            return self.neighbor(x)

        else:
            raise ValueError('Invalid Model_NUP forward mode!')


class A2C_Model_NUP(object):
    def __init__(self, config):
        self.model = Model_NUP(config=config)
        # states follow to states_
        self.states = None
        self.states_ = None

    def get_random_actions(self, dones):
        batch_size = len(self.states_)
        loader = DataLoader(self.states_, batch_size=batch_size, shuffle=False)
        for data in loader:
            prob_actions, critics = self.model('a2c', data.x, data.edge_index, data.batch)

        actions = []
        probs = [] 
        ACTION_DIM = []

        mark_length = 0
        for i in range(batch_size):
            len_state_x = len(self.states_[i].x)
            # get every state's action_vector index
            ACTION_DIM.append([mark_length + j for j in range(len_state_x)])
            mark_length += len_state_x

        # get every state's actions_probability
        temp_probs = [F.softmax(prob_actions[ACTION_DIM[i]], dim=0) for i in range(batch_size)]
        
        # select action via prob
        len_temp_probs = len(temp_probs)
        for p_num in range(len_temp_probs):
            if dones[p_num] != True: 
                action = np.random.choice(len(ACTION_DIM[p_num]), p=temp_probs[p_num].squeeze().cpu().data.numpy())
                actions.append(action)
                probs.append(temp_probs[p_num][action])
            else:
                action = -1
                actions.append(action)
                probs.append(1)

        return probs, critics, actions

    def get_greedy_actions(self, dones):
        batch_size = len(self.states_)
        loader = DataLoader(self.states_, batch_size=batch_size, shuffle=False)
        for data in loader:
            prob_actions, critics = self.model('a2c', data.x, data.edge_index, data.batch)

        actions = []
        probs = [] 
        ACTION_DIM = []

        mark_length = 0
        for i in range(batch_size):
            len_state_x = len(self.states_[i].x)
            ACTION_DIM.append([mark_length + j for j in range(len_state_x)])
            mark_length += len_state_x

        temp_probs = [F.softmax(prob_actions[ACTION_DIM[i]], dim=0) for i in range(batch_size)]

        len_temp_probs = len(temp_probs)
        for p_num in range(len_temp_probs):
            if dones[p_num] != True: 
                action_probs = temp_probs[p_num].squeeze().cpu().data.numpy()
                action = np.argmax(action_probs)
                all_index = []
                for possible_index in range(len(action_probs)):
                    if np.abs(action_probs[possible_index] - action_probs[action]) < 1e-4:
                        all_index.append(possible_index)
                action = min(all_index)
                actions.append(action)
                probs.append(temp_probs[p_num][action])
            else:
                action = -1
                actions.append(action)
                probs.append(1)

        return probs, critics, actions

    def update_embedding(self, infos=None):
        # infos: [(update_list, cat_vertex, tensor_index), (update_list, cat_vertex, tensor_index), ...]
        len_state_ = len(self.states_)
        if infos == None:
            for num in range(len_state_):
                self.states_[num].x = self.model('origin', self.states_[num].x)
        else:
            for num in range(len_state_):
                if infos[num][1] != -1:
                    # the state in environment and the state in a2c are independent
                    # states in a2c have embeddings with gradient, instead of states in environment
                    self.states_[num].x = self.states[num].x[infos[num][2]]
                    
                    target_cat = self.states[num].x[infos[num][1]]
                    for i in infos[num][0]:
                        temp_x = torch.cat((self.states_[num].x[i], target_cat))
                        self.states_[num].x[i] = self.model('neighbor', temp_x)
                else:
                    self.states_[num].x = self.states[num].x

    def follow(self):
        self.states = self.states_

    def reset(self):
        self.states = None
        self.states_ = None

    def parameters(self):
        return self.model.parameters()

    def to(self, device):
        self.model.to(device)

    def states_to(self, device):
        self.states_ = [state_.to(device) for state_ in self.states_]

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def save(self, name):
        torch.save(self.model.state_dict(), name + '.pth')
