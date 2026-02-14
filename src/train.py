import torch
from transformers import AutoModelForSequenceClassification 
from torchmetrics import Accuracy
from typing import Dict, Tuple, List
from tqdm import tqdm


def create_model(
    model_name: str, 
    num_labels: int, 
    device: torch.device
) -> torch.nn.Module:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        device_map="auto"
    ).to(device)
    return model

def train_step(
    model: torch.nn.Module,
    train_data: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module, 
    device: torch.device,
    num_classes: int,
    epoch: int,
    accum_steps: int = 0
) -> Tuple[float, float]:
    model.train()
    train_loss, train_acc = 0, 0

    for batch in train_data:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask=attention_mask).logits
        loss = loss_fn(logits, labels)
        train_loss += loss.item()
        loss.backward()
        if accum_steps > 0 and epoch % accum_steps != 0:
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        preds = torch.argmax(logits, dim=1)
        train_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        train_acc += train_accuracy(preds, labels).item()
    return train_loss / len(train_data), train_acc / len(train_data)

def val_step(
    model: torch.nn.Module,
    val_data: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    num_classes: int
) -> Tuple[float, float]:
    model.eval()
    val_loss, val_acc = 0, 0

    with torch.inference_mode():
        for batch in val_data:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask=attention_mask).logits
            loss = loss_fn(logits, labels)
            val_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            val_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
            val_acc += val_accuracy(preds, labels).item()
    return val_loss / len(val_data), val_acc / len(val_data)
        
def train_model(
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
    if accum_steps > 0 and accum_steps > epochs:
        raise ValueError("'accum_steps' must be lower than 'epochs'")
    results = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            train_data=train_data,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            num_classes=num_classes,
            epoch=epoch,
            accum_steps=accum_steps
        )
        val_loss, val_acc = val_step(
            model=model,
            val_data=val_data,
            loss_fn=loss_fn,
            device=device,
            num_classes=num_classes
        )
        print(f"{"Epoch:":<20} {epoch + 1:>10}\n"
              f"{"Train Loss:":<20} {train_loss:>10.3f}\n"
              f"{"Train Accuracy:":<20} {train_acc:>10.3f}\n"
              f"{"Val Loss:":<20} {val_loss:10.3f}\n"
              f"{"Val Accuracy:":<20} {val_acc:>10.3f}"
        )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
    return results