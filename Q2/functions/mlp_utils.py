from __future__ import annotations
from pathlib import Path
import torch
import numpy as np

def train(model, train_loader, optimizer, criterion, epoch, EPOCHS, device,
          log_interval=50, model_name=None, output_dir="outputs"):
    model.train()
    loss_cpu = 0.0
    correct = 0
    total = 0

    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data['x'], data['label']
        inputs, target = inputs.to(device), target.to(device)

        if inputs.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8):
            inputs = inputs.float()

        optimizer.zero_grad()

        outputs = model(inputs)                 
        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()

        loss_cpu += loss.item()

        predicted = outputs.argmax(dim=1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        if batch_idx % log_interval == 0:
            acc = 100.0 * correct / total if total > 0 else 0.0
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\tLoss: %.4f Acc@1: %.3f%%'
                  % (epoch, EPOCHS, batch_idx+1, len(train_loader),
                     loss.item(), acc))
        if model_name and epoch == EPOCHS:
                out_dir = Path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), out_dir / f"{model_name}.pt")

    return loss_cpu / len(train_loader), correct / total

def predict_proba(model, loader,device):
    model.eval()
    all_logits = []
    all_y = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            yb = batch["label"].cpu().numpy()

            logits = model(x).cpu().numpy()
            all_logits.append(logits)
            all_y.append(yb)

    logits = np.vstack(all_logits)
    y_true = np.concatenate(all_y)

    # Softmax -> probabilities
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)

    y_pred = probs.argmax(axis=1)
    return y_true, y_pred, probs

def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["label"].to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * y.size(0)
            total += y.size(0)
    return total_loss / total

best_val = float("inf")
patience = 5
wait = 0