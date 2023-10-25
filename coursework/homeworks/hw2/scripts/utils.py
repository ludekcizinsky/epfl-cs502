import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score
import time
# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, config=None):

    # Experiment tracking
    if config is not None:
        import wandb
        wandb.init(project="cs502-hw2-gnns", config=config)
        wandb.watch(model)

    # Store the train and val performance over epochs
    train_losses, train_f1s = [], []
    val_losses, val_f1s = [], []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        y_true, y_pred = torch.empty((0,)), torch.empty((0,))
        start_time = time.time()

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            logits = torch.sigmoid(outputs)
            predicted = (logits > 0.5).int()  # Assuming a threshold of 0.5 for binary classification

            y_true = torch.cat((y_true, labels))
            y_pred = torch.cat((y_pred, predicted))

        end_time = round(time.time() - start_time, 2)
        train_f1 = f1_score(y_true, y_pred, average='macro')
        train_losses.append(total_loss)
        train_f1s.append(train_f1)

        # Evaluate on the validation set
        model.eval()
        total_loss_val = 0.0
        y_true_val, y_pred_val = torch.empty((0,)), torch.empty((0,))
        with torch.no_grad():
            for inputs_val, labels_val in val_loader:
                outputs_val = model(inputs_val)
                loss_val = criterion(outputs_val, labels_val)
                total_loss_val += loss_val.item()
                logits_val = torch.sigmoid(outputs_val)
                predicted_val = (logits_val > 0.5).int()

                y_true_val = torch.cat((y_true_val, labels_val))
                y_pred_val = torch.cat((y_pred_val, predicted_val))

        val_f1 = f1_score(y_true_val, y_pred_val, average='macro')
        val_losses.append(total_loss_val)
        val_f1s.append(val_f1)

        # Log metrics to wandb
        if config is not None:
            wandb.log({"Train Loss": total_loss, "Train F1 (macro)": train_f1,
                    "Validation Loss": total_loss_val, "Validation F1 (macro)": val_f1})

        print(f'Epoch [{epoch + 1}/{num_epochs}]({end_time}s) '
              f'Train Loss: {total_loss:.4f} Train F1(macro): {train_f1:.2f}% '
              f'Validation Loss: {total_loss_val:.4f} Validation F1(macro): {val_f1:.2f}%')
         
    return (train_losses, train_f1s, val_losses, val_f1s)

def plot_losses_and_f1s(train_losses, train_f1s, val_losses, val_f1s):
    fig, axs = plt.subplots(figsize=(12, 6), nrows=1, ncols=2)

    # Loss
    sns.lineplot(x=range(len(train_losses)), y=train_losses, label='Train Loss', ax=axs[0])
    sns.lineplot(x=range(len(val_losses)), y=val_losses, label='Validation Loss', ax=axs[0])

    # F1
    sns.lineplot(x=range(len(train_f1s)), y=train_f1s, label='Train F1', ax=axs[1])
    sns.lineplot(x=range(len(val_f1s)), y=val_f1s, label='Validation F1', ax=axs[1])

    return fig

# Define the evaluation function
def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predicted = (outputs > 0.5).int()  # Assuming a threshold of 0.5 for binary classification
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    print(f'Test Loss: {total_loss:.4f} Test Accuracy: {accuracy:.2f}%')