import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score
import time

# Define the training loop
def train(model, train_loader, criterion, optimizer, num_epochs=10):
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
            logits = F.sigmoid(outputs)
            predicted = (logits > 0.5).int()  # Assuming a threshold of 0.5 for binary classification

            y_true = torch.cat((y_true, labels))
            y_pred = torch.cat((y_pred, predicted))

        end_time = round(time.time() - start_time, 2)
        f1 = f1_score(y_true, y_pred, average='macro')
        print(f'Epoch [{epoch + 1}/{num_epochs}]({end_time}s) Loss: {total_loss:.4f} F1(macro): {f1:.2f}%')

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