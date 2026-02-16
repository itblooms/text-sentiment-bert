A pet project for fine-tuning DistilBETR model from 🤗 Hugging Face Transformers to perform semantic classification task. 
Project aim was to create scripts that take YAML config, which describes detailes of trainig and based on it run multiple experiments to find optimal hyperparameters

## Run tuning
To run a tuning script simply run 
```bash
python -m experiments.exp --config path/to/config
```
The results of a training can be analysed using MLflow by running
```bash
mlflow ui
```
and proceeding to the link it provides.
## Config
An example of a config file for LoRA fine-tuning
```yaml
experiment_name: peft_lora

model:
  name: distilbert/distilbert-base-uncased
  num_labels: 6

dataset: dair-ai/emotion

search_space:
  epochs: [1, 3]
  lr: [1.0e-5, 1.0e-3]
  batch_size: [8, 16, 32]
  r: [2, 4, 8, 16, 32]
```

It is possible to choose any dataset you want, that you can find on 🤗 Hugging Face Hub or create your own using 🤗 Datasets.
Current implementation suggest that all parameters in `search_space`, except `lr` are randomly chosen out of provided variants, while `lr` is sampled from loguniform distribution with provided bounds.
