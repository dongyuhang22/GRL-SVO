# GRL-SVO

## Overview

Implementation of [Suggesting Variable Order for Cylindrical Algebraic Decomposition via Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/efcb5b06ce8bb672ffa26b9dc5cdd0f9-Abstract-Conference.html), NeurIPS 2023.

GRL-SVO uses reinforcement learning to choose a variable ordering for cylindrical algebraic decomposition (CAD). It provides two variants: **UP** updates the polynomial system through projection as it selects variables, while **NUP** uses learned graph updates without computing projections.

## Installation

Python 3.10+ on Linux. Run commands from the repository root:

```bash
pip install -r requirements.txt
```

For UP prediction, replace `/path/to/maple` with the Maple command-line executable. UP training uses cached projections.

## Prediction and evaluation

Run the published checkpoints on the random three-variable test set:

```bash
python predict.py --variant nup --output outputs/nup.jsonl
python predict.py --variant up --maple /path/to/maple --output outputs/up.jsonl
python evaluate.py outputs/nup.jsonl
python evaluate.py outputs/up.jsonl
```

For all bundled NUP test sets, add `--variables 3 4 5 6 7 8 9` to the prediction command. Evaluation uses the bundled CAD results.

For your own polynomials, use [data/example.jsonl](data/example.jsonl) as a template, with one JSON object per line:

```bash
python predict.py --variant nup --input data/example.jsonl --output outputs/example-nup.jsonl
python predict.py --variant up --input data/example.jsonl --maple /path/to/maple --output outputs/example-up.jsonl
```

## Training

Train on the bundled three-variable data for 100 epochs, with validation after every epoch:

```bash
python train.py --variant nup --output outputs/train-nup
python train.py --variant up --output outputs/train-up
```

Each run saves `best.pth` (best validation checkpoint) and `last.pth` (final epoch). To test either file, add `--checkpoint /path/to/model.pth` to the matching prediction command above.

Use a new output file or directory for each run.

## Acknowledgements

`projection.mpl` adapts nine routines from [ProjectionCAD V3.18 (11 February 2015)](https://researchdata.bath.ac.uk/71/), maintained by Matthew England at the University of Bath, with Russell Bradford, James H. Davenport and David Wilson. University of Bath is the rights holder. The GRL-SVO adaptation extracts the projection/factorization routines, exposes two entry points, renames the module and removes documentation. This adapted file is distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Contact

- Fuqi Jia: [jiafq@ios.ac.cn](mailto:jiafq@ios.ac.cn)
- Yuhang Dong: [dongyuhang22@mails.ucas.ac.cn](mailto:dongyuhang22@mails.ucas.ac.cn)
