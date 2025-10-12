import torch
from transformers import AutoModelForSequenceClassification 
from torchmetrics import Accuracy
from typing import Dict, Tuple, List
from tqdm import tqdm


def create_model(model_name: str, 
                 num_labels: int, 
                 device: torch.device) -> torch.nn.Module:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    ).to(device)
    return model

def train_step(model: torch.nn.Module,
               train_data: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.Module, 
               device: torch.device,
               config) -> Tuple[float, float]:
    
    model.train()
    train_loss, train_acc = 0, 0

    for batch in train_data:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask=attention_mask).logits
        loss = loss_fn(logits, labels)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        train_accuracy = Accuracy(task="multiclass", 
                             num_classes=config.model["num_classes"])
        train_acc += train_accuracy(preds, labels)

    return train_loss / len(train_data), train_acc / len(train_data)

def val_step(model: torch.nn.Module,
             val_data: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device: torch.device,
             config) -> Tuple[float, float]:
    
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
            val_accuracy = Accuracy(task="multiclass", 
                               num_classes=config.model["num_classes"])
            val_acc += val_accuracy(preds, labels)

    return val_loss / len(val_data), val_acc / len(val_data)
        
def train_model(model: torch.nn.Module,
                train_data: torch.utils.data.DataLoader,
                val_data: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                loss_fn: torch.nn.Module,
                device: torch.device,
                epochs: int,
                config) -> Dict[str, List[float]]:
        
    results = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           train_data=train_data,
                                           optimizer=optimizer,
                                           loss_fn=loss_fn,
                                           device=device,
                                           config=config)
        val_loss, val_acc = val_step(model=model,
                                     val_data=val_data,
                                     loss_fn=loss_fn,
                                     device=device,
                                     config=config)

        print(f"Epoch:           | {epoch + 1}\n"
              f"Train Loss:      | {train_loss:.3f}\n"
              f"Train Accuracy:  | {train_acc:.3f}\n"
              f"Val Loss:        | {val_loss:.3f}\n"
              f"Val Accuracy:    | {val_acc:.3f}")
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
    return results
