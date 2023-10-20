class EnvT_config(object):
    def __init__(self, 
                 var=3,
                 file=None, 
                 mode=None, 
                 norm_mode=None,
                 max_cells=None,
                 have_embedding=True,
                 have_edge=True,
                 feature_mode='train') -> None:

        self.var = var
        self.file = file
        self.mode = mode
        self.norm_mode = norm_mode
        self.max_cells = max_cells
        self.have_embedding = have_embedding
        self.have_edge = have_edge
        self.feature_mode = feature_mode

    def change(self, 
               file, 
               mode):

        self.file = file
        self.mode = mode


class EnvP_config(object):
    def __init__(self, 
                 file=None, 
                 dataset=None, 
                 norm_mode=None,
                 step=None,
                 remain_time=None,
                 model_dir = None,
                 have_embedding=True,
                 have_edge=True,
                 feature_mode='train') -> None:

        self.file = file
        self.dataset = dataset
        self.norm_mode = norm_mode
        self.step = step
        self.remain_time = remain_time
        self.model_dir = model_dir
        self.have_embedding = have_embedding
        self.have_edge = have_edge
        self.feature_mode = feature_mode        

    def change(self, 
               file,
               remain_time):

        self.file = file
        self.remain_time = remain_time


class model_nup_config(object):
    def __init__(self, 
                 num_convs, 
                 hidden_channels, 
                 actor_mlp_layers, 
                 actor_mlp_list, 
                 critic_mlp_layers, 
                 critic_mlp_list,                 
                 origin_mlp_layers, 
                 origin_mlp_list, 
                 neighbor_mlp_layers, 
                 neighbor_mlp_list,
                 conv='GRAPH') -> None:

        self.conv = conv                 
        self.num_convs = num_convs
        self.hidden_channels = hidden_channels
        self.actor_mlp_layers = actor_mlp_layers
        self.actor_mlp_list = actor_mlp_list
        self.critic_mlp_layers = critic_mlp_layers
        self.critic_mlp_list = critic_mlp_list
        self.origin_mlp_layers = origin_mlp_layers
        self.origin_mlp_list = origin_mlp_list
        self.neighbor_mlp_layers = neighbor_mlp_layers
        self.neighbor_mlp_list = neighbor_mlp_list


class model_config(object):
    def __init__(self, 
                 num_convs, 
                 hidden_channels, 
                 actor_mlp_layers, 
                 actor_mlp_list, 
                 critic_mlp_layers, 
                 critic_mlp_list,
                 conv='GRAPH') -> None:
                 
        self.conv = conv
        self.num_convs = num_convs
        self.hidden_channels = hidden_channels
        self.actor_mlp_layers = actor_mlp_layers
        self.actor_mlp_list = actor_mlp_list
        self.critic_mlp_layers = critic_mlp_layers
        self.critic_mlp_list = critic_mlp_list
