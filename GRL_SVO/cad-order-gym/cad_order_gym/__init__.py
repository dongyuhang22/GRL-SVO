from gym.envs.registration import register

register(
    id='cad_order_train_up-v0',                                   # Format should be xxx-v0, xxx-v1....
    entry_point='cad_order_gym.envs:CADEnvTUP',              # Expalined in envs/__init__.py
)
register(
    id='cad_order_pred_up-v0',                                   # Format should be xxx-v0, xxx-v1....
    entry_point='cad_order_gym.envs:CADEnvPUP',              # Expalined in envs/__init__.py
)
register(
    id='cad_order_train_nup-v0',                                   # Format should be xxx-v0, xxx-v1....
    entry_point='cad_order_gym.envs:CADEnvTNUP',              # Expalined in envs/__init__.py
)
register(
    id='cad_order_pred_nup-v0',                                   # Format should be xxx-v0, xxx-v1....
    entry_point='cad_order_gym.envs:CADEnvPNUP',              # Expalined in envs/__init__.py
)