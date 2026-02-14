import os
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
import mlflow
import copy
import ray
from ray import tune
from ray.tune import Tuner, TuneConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.search.optuna import OptunaSearch
from argparse import ArgumentParser
from src.train import create_model, train_model
from src.data import load_data
from typing import Dict, Any, List, Optional
from peft import get_peft_model, LoraConfig, TaskType


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

def train_lora(
    model: torch.nn.Module,
    train_data: torch.utils.data.DataLoader,
    val_data: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    epochs: int,
    num_classes: int,
    lora_r: int,
    accum_steps: int = 0
):
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=lora_r,
        lora_alpha=2*lora_r,
        lora_dropout=0.1,
        init_lora_weights="gaussian",
        target_modules=["q_lin", "v_lin"]
    )
    lora_model = get_peft_model(model, lora_config) # type: ignore
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
    base_cfg: Dict[str, Any],
    model: torch.nn.Module, 
    loss_fn: torch.nn.Module,
    device: torch.device,
    num_workers: int,
    accum_steps: int = 0,
    ) -> None:
    model = copy.deepcopy(model).to(device)
    train_loader, val_loader, _, _ = load_data(
        model_name=base_cfg["model"]["name"],
        dataset_name=base_cfg["dataset"],
        batch_size=config["batch_size"],
        num_workers=num_workers
    )
    optimizer = Adam(params=model.parameters(), lr=config["lr"])

    mlflow.set_experiment(base_cfg["experiment_name"])
    run_name = "_".join(f"{k}={v}" for k, v in config.items())
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(config)
        match base_cfg["experiment_name"]:
            case "fft":
                results = fft_train(
                    model=model, train_data=train_loader, val_data=val_loader,
                    optimizer=optimizer, loss_fn=loss_fn, device=device, 
                    epochs=config["epochs"], num_classes=base_cfg["model"]["num_classes"],
                    accum_steps=accum_steps
                )
            case "peft_head_n_layers":
                results = train_head_n_layers(
                    model=model, train_data=train_loader, val_data=val_loader,
                    optimizer=optimizer, loss_fn=loss_fn, device=device, 
                    epochs=config["epochs"], num_classes=base_cfg["model"]["num_classes"],
                    n_layers=config["n_layers"], accum_steps=accum_steps
                )
            # case "peft_adapter":
                # results = train_adapter
            case "peft_lora":
                results = train_lora(
                    model=model, train_data=train_loader, val_data=val_loader,
                    optimizer=optimizer, loss_fn=loss_fn, device=device,
                    epochs=config["epochs"], num_classes=base_cfg["model"]["num_labels"],
                    lora_r=config["r"], accum_steps=accum_steps
                )
            case _:
                raise ValueError("Unsupported fine-tuning method. Please use one of these "
                                 "'fft', 'peft/head_n_layers', 'peft/adapter', 'peft/lora'")
        mlflow.log_metric("train_loss", results["train_loss"][-1])
        mlflow.log_metric("val_accuracy", results["val_acc"][-1])
        tune.report({"val_accuracy": results["val_acc"][-1]})

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 1  # os.cpu_count() if os.cpu_count() is not None else 0 
    model = create_model(config["model"]["name"], config["model"]["num_labels"], device)
    loss_fn = nn.CrossEntropyLoss()

    ray.init(num_gpus=1)
    search_space = parse_search_space(config["search_space"])
    optuna_search = OptunaSearch()
    tuner = Tuner(
        tune.with_resources(
            tune.with_parameters(
                train_fn,
                base_cfg=config,
                model=model,
                loss_fn=loss_fn,
                device=device,
                num_workers=num_workers
            ),
            resources={"cpu": 4, "gpu": 1}
        ),
        param_space=search_space,
        tune_config=TuneConfig(
            search_alg=optuna_search,
            metric="val_accuracy", 
            mode="max",
            num_samples=10,
            max_concurrent_trials=3
        )
    )
    tuner.fit()

if __name__ == "__main__":
    main()