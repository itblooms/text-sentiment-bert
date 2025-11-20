import os
import torch
import torch.nn as nn
from torch.optim import Adam
import mlflow
import copy
from ray import tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from omegaconf import OmegaConf, DictConfig
from argparse import ArgumentParser
from src.train import create_model, train_model
from src.data import load_data
from typing import Dict, Any, List, Optional

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

def train_head_n_layers(
    model: torch.nn.Module, 
    train_data: torch.utils.data.DataLoader,
    val_data: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    epochs: int,
    num_classes: int,
    n_layers: int,
    accum_steps: int = 0
):
    for param in model.classifier.parameters(): # type: ignore
        param.requires_grad = True
    for param in model.pre_classifier.parameters(): # type: ignore
        param.requires_grad = True
    n_blocks = len(model.distilbert.transformer.layer) # type: ignore
    for i in range(1,  n_layers + 1):
        for param in model.distilbert.transformer.layer[n_blocks-i].parameters(): # type: ignore
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
    
def train_fn(
    config: Dict[str, Any],
    *,
    base_cfg: DictConfig,
    model: torch.nn.Module, 
    loss_fn: torch.nn.Module,
    device: torch.device,
    num_workers: int,
    accum_steps: int = 0,
    ) -> None:
    supported_methods = ["fft", "peft_lora", "peft_adapter", "peft_head_n_layers"]
    if base_cfg.experiment_name not in supported_methods:
            raise ValueError("Unsupported fine-tuning method. Please use one of these "
                             "'fft', 'peft/head_n_layers', 'peft/adapter', 'peft/lora'")
    model = copy.deepcopy(model)
    train_loader, val_loader, _, _ = load_data(
        model_name=base_cfg.model["name"], # type: ignore
        dataset_name=base_cfg.dataset,
        batch_size=config["batch_size"],
        num_workers=num_workers # type: ignore
    )
    optimizer = Adam(params=model.parameters(), lr=config["lr"])

    mlflow.set_experiment(base_cfg.experiment_name)
    run_name = "_".join(f"{k}={v}" for k, v in config.items())
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(config)
        match base_cfg.experiment_name:
            case "fft":
                results = fft_train(
                    model=model, train_data=train_loader, val_data=val_loader,
                    optimizer=optimizer, loss_fn=loss_fn, device=device, 
                    epochs=config["epochs"], num_classes=base_cfg.model["num_classes"],
                    accum_steps=accum_steps
                )
            case "peft_head_n_layers":
                results = train_head_n_layers(
                    model=model, train_data=train_loader, val_data=val_loader,
                    optimizer=optimizer, loss_fn=loss_fn, device=device, 
                    epochs=config["epochs"], num_classes=base_cfg.model["num_classes"],
                    n_layers=config["n_layers"], accum_steps=accum_steps
                )
            case "peft_adapter":
                results = train_adapter
            case "peft_lora":
                results = train_lora


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type="str")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = os.cpu_count() if os.cpu_count() is not None else 0 
    model = create_model(config.model["name"], config.model["num_labels"], device)
    loss_fn = nn.CrossEntropyLoss()

    search_space = parse_search_space(config.search_space)
    tuner = tune.Tuner(
        tune.with_parameters(
            train_fn,
            base_cfg=config,
            model=model,
            loss_fn=loss_fn,
            device=device,
            num_workers=num_workers
        ),
        param_space=search_space
    )
    tuner.fit()