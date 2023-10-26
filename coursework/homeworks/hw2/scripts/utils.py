"""Collection of utility functions for training and evaluation of 
various configurations of GNNs on the MUTAG dataset.
"""

# ---------------- Import libraries and/or modules
# PyTorch
import torch

# Sklearn
from sklearn.metrics import f1_score, accuracy_score

# Bulit-in
import time

# Plotting/Pretty Printing
import seaborn as sns
import networkx as nx
from rich import print

# Built in
from IPython.display import clear_output

# ---------------- Training Utilities
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, config=None, verbosity=1):
    """Train and evaluate a model on the training and validation sets.
    Args:
        model (torch.nn.Module): model to be trained and evaluated.
        train_loader (torch.utils.data.DataLoader): data loader for the training set.
        val_loader (torch.utils.data.DataLoader): data loader for the validation set.
        criterion (torch.nn.Module): loss function.
        optimizer (torch.optim.Optimizer): optimizer.
        num_epochs (int): number of epochs to train the model for.
        config (dict): configuration parameters for the experiment. (optional)
        verbosity (int): level of verbosity. (optional)
    
    Returns:
        train_losses (list): training losses over epochs.
        train_f1s (list): training F1 scores over epochs.
        val_losses (list): validation losses over epochs.
        val_f1s (list): validation F1 scores over epochs.
    """

    # (optional) Experiment tracking using wandb
    if config is not None:
        import wandb
        wandb.init(project="cs502-hw2-gnns", config=config)
        wandb.watch(model)

    # Store the train and val performance over epochs
    train_losses, train_f1s = [], []
    val_losses, val_f1s = [], []

    # Train the model
    for epoch in range(num_epochs):

        # Turn on training mode
        model.train()

        # Setup variables to accumulate statistics about
        total_loss = 0.0
        y_true, y_pred = torch.empty((0,)), torch.empty((0,))
        start_time = time.time()

        # Iterate over the batch
        for inputs, labels in train_loader:

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)

            # Loss computation
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Loss and prediction tracking
            total_loss += loss.item()
            logits = torch.sigmoid(outputs)
            # Assumes a standard threshold of 0.5 for binary classification
            # under the assumption of 0-1 loss
            predicted = (logits > 0.5).int() 

            # Save the labels and predictions for the given batch
            y_true = torch.cat((y_true, labels))
            y_pred = torch.cat((y_pred, predicted))

        # Compute the average loss and F1 score over the epoch along with time taken
        end_time = round(time.time() - start_time, 2)
        train_f1 = f1_score(y_true, y_pred, average='macro')
        train_losses.append(total_loss)
        train_f1s.append(train_f1)

        # Evaluate on the validation set
        total_loss_val, val_f1, _, _ = evaluate(model, val_loader, criterion)
        val_losses.append(total_loss_val)
        val_f1s.append(val_f1)

        # (optional) Log metrics to wandb
        if config is not None:
            wandb.log({"Train Loss": total_loss, "Train F1 (macro)": train_f1,
                    "Validation Loss": total_loss_val, "Validation F1 (macro)": val_f1})

        if verbosity > 0:
            clear_output(wait=True)
            print(f'Epoch [{epoch + 1}/{num_epochs}]({end_time} s)\n'
            f'\tTrain Loss: [bold]{total_loss:.4f}[/bold] Train F1(macro): [bold]{train_f1:.2f}[/bold]\n'
            f'\tValidation Loss: [bold]{total_loss_val:.4f}[/bold] Validation F1(macro): [bold]{val_f1:.2f}[/bold]')
         
    return train_losses, train_f1s, val_losses, val_f1s

# ---------------- Evaluation Utilities
def evaluate(model, test_loader, criterion):
    """Evaluate a model on the test/validation set.

    Args:
        model (torch.nn.Module): model to be evaluated.
        test_loader (torch.utils.data.DataLoader): data loader for the test/validation set.
        criterion (torch.nn.Module): loss function.
    Returns:
        total_loss (float): total loss over the test/validation set.
        eval_f1 (float): Macro F1 score over the test/validation set. 
    """

    # Set the model to evaluation mode
    model.eval()

    # Keep track of the loss and predictions
    total_loss = 0.0
    y_true, y_pred = torch.empty((0,)), torch.empty((0,))

    # Don't compute gradients
    with torch.no_grad():
        
        # Batch iteration
        for inputs, labels in test_loader:

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Compute the predictions
            logits = torch.sigmoid(outputs)
            predicted = (logits > 0.5).int()

            # Save the predictions and labels
            y_true = torch.cat((y_true, labels))
            y_pred = torch.cat((y_pred, predicted))

    # Compute the macro F1 score
    eval_f1 = f1_score(y_true, y_pred, average='macro')
    eval_acc = accuracy_score(y_true, y_pred)

    return total_loss, eval_f1, eval_acc, (y_true, y_pred)

# ---------------- Plotting Utilities
def plot_losses_and_f1s(losses, f1s, axs, label):
    """Plot the losses and F1 scores over epochs.
    Args:
        losses (list): losses over epochs.
        f1s (list): F1 scores over epochs.
        ax (matplotlib.axes.Axes): axes to plot on.
    """
    sns.lineplot(x=range(len(losses)), y=losses, ax=axs[0], label=f"{label} Loss")
    sns.lineplot(x=range(len(f1s)), y=f1s, ax=axs[1], label=f"{label} F1 (macro)")

def plot_graph(edge_index, ax, node_colors, edge_colors):
    """Plots the given graph.

    Args:
        edge_index (torch.Tensor): edge index of a graph.
        ax (matplotlib.axes.Axes): Axis object where the graph should be plotted.
    Returns:
        G (networkx.Graph): NetworkX graph.
    """

    # Create an empty NetworkX graph
    G = nx.Graph()

    # Cast the edge index to int
    edge_index = edge_index.int()

    # Extract source and target nodes from the edge_index
    source_nodes, target_nodes = edge_index[:, 0].tolist(), edge_index[:, 1].tolist()

    # Add nodes to the graph
    nodes = set(source_nodes + target_nodes)
    G.add_nodes_from(nodes)

    # Create a list of edges as 2-tuples
    edges = [(source, target) for source, target in zip(source_nodes, target_nodes)]

    # Add edges to the graph
    i = 0
    for s, t in edges:
        G.add_edge(s, t, color=edge_colors[i])
        i += 1

    # Plot the graph on the specified axis
    pos = nx.kamada_kawai_layout(G)  # Define the layout algorithm
    nx.draw(G, pos, with_labels=True, node_size=500, node_color=node_colors, font_weight='bold', font_color='white', font_size=10, ax=ax)

    # Manually draw the edges with custom colors
    for edge, color in zip(G.edges, edge_colors):
        nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color=color, width=2.0, ax=ax)

def get_feat_colors(feat, cmap):
    """Get colors for the nodes/edges of a graph based on their features.

    Args:
        feat (torch.Tensor): features of the nodes/edges of a graph.
        cmap (matplotlib.colors.ListedColormap): colormap.

    Returns:
        colors (list): list of colors for the nodes/edges of a graph. 
    """

    n, m = feat.shape
    colors = [cmap(i / m) for i in range(n)]

    return colors