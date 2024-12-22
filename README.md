## ML-based-method-for-identifying-Industrial-Symbiosis-opportunties

### Accompanying code for the paper "A novel Machine Learning-based method for identifying Industrial Symbiosis opportunities"

This repository contains:
- model architecture
- model training
- model performance evaluation
- LLMs evaluation and comparison
- training output files (including pre-trained weights)


### 1. Data preparation

Our data come from [EcoInvent](https://ecoinvent.org/), and we are unable to share the data due to licensing restrictions.
Therefore, we provide the instruction here on how to prepare the data in a suitable format for training the models.

- prepare a `csv` file involving three columns: activity, waste and label (`produce` / `need`)
- need to include negative samples as well (label = `neither`)
- this file include all data (both produce and need labels, before train-val-test splitting)

### 2. Model training and evaluation

Train the model and evaluate the model.
```shell
python train.py --EI_file data.csv --output_dir output_new/train_produce_v2 --epochs 40 --dataset_mode produce --embedding_model GPT2 --lr 0.001 --l2_lambda 0
```

Evaluate a trained model.
```shell
python train.py --eval_only --load_pretrained output_new/train_produce_v2/model_state_dict.pth --EI_file data.csv --output_dir output_new/train_produce_v2 --dataset_mode produce --embedding_model GPT2
```
Note: if want to evaluate on other datasets / with more complex settings, go to `evaluation.py` to customize the code. 

Arguments:
- `EI_file`: the csv file that just prepared
- `dataset_mode`: whether to train Produce Relationship (PR) or Need Relationship (NR) model (the code will automatically filter corresponding labels and split into train/val/test)
- `embedding_model`: the choice of PLM, could be GPT2 / E5 / Nomic / Jina
- `lr`: learning rate
- `l2_lambda`: coefficient for L2 regularization
- `train_slice`: sampling rate for training models with less data (sensitivity analysis), default is `1`
- `train_slice_seed`: random seed for the train set sampling
- `load_pretrained`: the pretrained model weight (for continue training or evaluation purposes)
- `eval_only`: only performance evaluation
- more...

input file:
- the csv file just prepared

output files:
- model weights in `pth` format
- plot for step loss
- ROC for each set
- confusion matrix for each set
- more ...

### LLMs evaluation

Evaluate the performances of LLMs on the same test set.

```shell
python prompt_LLM.py --model llama3.1 --EI_file data.csv --output_dir output_LLM --dataset_mode produce
```

Arguments:
- `model`: support llama3.1, gpt series (need to download llama model or set up OpenAI AIP keys)
- `EI_file`: the csv data file (the code will automatically create the same test set as training ML models)
- `dataset_mode`: whether to predict produce or need relationships

input file:
- the csv file just prepared

output files:
- a csv file with LLM response recorded for each data sample
- resulting metrics