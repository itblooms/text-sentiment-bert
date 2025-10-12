import os
import torch
import torch.nn as nn
from torch.optim import Adam
import mlflow
from ray import tune
from omegaconf import OmegaConf
import argparse
from src.train import create_model, train_model
from src.data import load_data


def fft_train(
        model: torch.nn.Module, 
        train_data: torch.utils.data.DataLoader,
        val_data: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        device: torch.device,
        epochs: int,
        num_classes: int
) -> None:
    for param in model.parameters():
        param.requires_grad = True
    train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=epochs,
        num_classes=num_classes
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type="str")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = os.cpu_count() if os.cpu_count() is not None else 0
    train_loader, val_loader, _, _ = load_data(
        model_name=config.data["name"],
        dataset_name=config.model["name"],
        batch_size=config.train["batch_size"],
        num_workers=cores if (cores := os.cpu_count()) is not None else 0
    )
    model = create_model(config.model["name"], config.data["num_labels"], device)
    optimizer = Adam(params=model.parameters(), lr=config.train["lr"])
    loss_fn = nn.CrossEntropyLoss()
    fft_train(
        model=model,
        train_data=train_loader,
        val_data=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=config.train["epochs"],
        num_classes=config.data["num_classes"]
    )