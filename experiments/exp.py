import os
import torch
import torch.nn as nn
from torch.optim import Adam
import mlflow
from ray import tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from omegaconf import OmegaConf
from argparse import ArgumentParser
from src.train import create_model, train_model
from src.data import load_data
from typing import Dict, Any, List

def parse_search_space(ss_dict: Dict[str, Any]) -> Dict[str, Any]:
    search_space = {}
    for k, v in ss_dict.items():
        if isinstance(v, list):
            if k in ["epochs", "batch_size", "r", "hidden_dim", "n_layers"]:
                search_space[k] = tune.choice(v)
            elif k in ["lr"]:
                search_space[k] = tune.loguniform(*v)
        else:
            search_space[k] = v
    return search_space

def fft_train(
    model: torch.nn.Module, 
    train_data: torch.utils.data.DataLoader,
    val_data: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    epochs: int,
    num_classes: int,
    accum_steps: int = 0
) -> Dict[str, List[float]]:
    for param in model.parameters():
        param.requires_grad = True
    results = train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=epochs,
        num_classes=num_classes,
        accum_steps=accum_steps
    )
    return results

def train_lora():
    pass

def train_adapter():
    pass

def train_head_n_layers():
    pass

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type="str")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = os.cpu_count() if os.cpu_count() is not None else 0

    train_loader, val_loader, _, _ = load_data(
        model_name=config.data["name"],
        dataset_name=config.model["name"],
        batch_size=config.train["batch_size"],
        num_workers=num_workers # type: ignore
    )
    model = create_model(config.model["name"], config.data["num_labels"], device)
    optimizer = Adam(params=model.parameters(), lr=config.train["lr"])
    loss_fn = nn.CrossEntropyLoss()

    match config.experiment_name:
        case "fft":
            train_fn = fft_train
        case "peft_head_n_layers":
            train_fn = train_head_n_layers
        case "peft_adapter":
            train_fn = train_adapter
        case "peft_lora":
            train_fn = train_lora
        case _:
            raise ValueError("Unsupported fine-tuning method. Please use one of these "
                             "'fft', 'peft/head_n_layers', 'peft/adapter', 'peft/lora'")
    search_space = {
        "experiment_name": config.experiment_name,
        "model_name": config.model_name,
        "train_fn": train_fn
    }

    tuner = tune.Tuner(
        tune.with_parameters(train_fn),
        param_space=search_space
    )
