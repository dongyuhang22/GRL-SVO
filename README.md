# GRL-SVO

## Setup

- GRL-SVO: source code and datasets;
- cad.tar.gz: conda environment;
- requirements.txt: required python packages.

#### Auto pip install

1. >conda create -n cad python=3.7.0
2. >conda activate cad
3. >pip install -r requirements.txt
4. >cd GRL-SVO/cad_order_gym
5. >pip install -e .
---

## Training

Position: `GRL-SVO/`

For GRL-SVO(NUP)
> python train_grl_svo_nup.py

For GRL-SVO(UP)
> python train_grl_svo_up.py

## Testing

Position: `GRL-SVO/`

For GRL-SVO(NUP)
> python predict_grl_svo_nup.py

The result is stored in `./predict_result/nup_rand3/result_nup.log`

> cat ./predict_result/nup_rand3/result_nup.log

For GRL-SVO(UP)
> python predict_grl_svo_up.py

GRL-SVO(UP) must interact with MAPLE during the testing process. If MAPLE is not installed, `predict_grl_svo_up.py` can only run one step, and the intermediate information stores in `GRL_SVO/predict_result/up_rand3`. 

> cat GRL_SVO/predict_result/up_rand3/action_1.log
