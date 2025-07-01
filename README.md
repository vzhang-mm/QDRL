# QDRL 6.18

This repository contains code for training and evaluating recurrent relational graph convolutional networks (RGCN) on temporal knowledge graph data. Example datasets included are `GDELT`, `ICEWS05-15`, `ICEWS14`, `ICEWS18`, and `YAGO`.

## Directory structure

```
QDRL-6.18/
├── main.py            # entry point for training/testing
├── opt.py             # command line options
├── src/               # model components
└── data/
    ├── GDELT/
    ├── ICEWS05-15/
    ├── ICEWS14/
    ├── ICEWS18/
    └── YAGO/
```
### Datasets

Each dataset directory contains files such as `train.txt`, `valid.txt`, `test.txt`, `entity2id.txt` and `relation2id.txt`. Each line of the train/valid/test files lists a triplet or quadruple depending on whether timestamps are used.

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org) (tested with 1.13)
- [DGL](https://www.dgl.ai) (tested with 1.1)
- `numpy`, `tqdm`

Install the dependencies with pip:

```bash
pip install torch dgl numpy tqdm
```

## Training

Training is performed with `main.py`.  The script automatically constructs a model directory in `./models/` and saves checkpoints there.  A typical command for entity prediction is:

```bash
python main.py -d <DATASET> \
    --train-history-len 7 \
    --lr 0.0005 \
    --n-layers 2 \
    --gpu 0 \
    --n-hidden 256 \
    --decoder convtranse \
    --encoder uvrgcn \
    --layer-norm \
    --pre-weight 0.7 \
    --pre-type TF \
    --add-static-graph
```

For **relation prediction** add `--relation-prediction`:

```bash
python main.py -d <DATASET> --relation-prediction [other options]
```

The training script will periodically evaluate on the validation set and save the best model to `./models/`.

## Inference / Testing

To evaluate a trained model on the test set, pass the `--test` flag with the same configuration used during training.  `main.py` will load the corresponding checkpoint from `./models/` if it exists:

```bash
python main.py -d <DATASET> --test [same options as training]
```

During testing the script prints MRR and Hits@K metrics and also writes a CSV summary under `./result/`.

## Notes

- `--add-his-graph` enables the use of preprocessed historical subgraphs if available.
- Model and result directories are created automatically when running the script.

