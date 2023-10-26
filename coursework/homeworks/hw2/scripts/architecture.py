"""This module provides high level abstraction that based on provided
architecture and pooling function constructs a full Graph Neural Network.
"""

# ---------------- Import libraries and/or modules
from torch import nn
from torch.nn import functional as F
import torch

# ---------------- Graph Neural Network
class GNN(nn.Module):
    """A full Graph Neural Network for mutagenicity prediction."""

    def __init__(self, architecture, pooling, dropout=0.1):
        """
        Initialize the network.
        Args:
            architecture (list of lists): 
                List of layers, each layer is a list of the given layer type and arguments for the layer
                passed as a dictionary.
            pooling (nn.Module or callable):
                Pooling function to apply, as x_pooled = pooling(x).
            dropout (float):
                Dropout probability. (optional)
        """
        # Initialize the parent class
        super().__init__() 

        # Save the dropout probability
        self.dropout = dropout
        
        # Deep Graph Convolutional Architecture
        self.convs = nn.ModuleList()
        self.conv_dims = []
        for layer in architecture:
            # Add the layer
            self.convs.append(layer[0](**layer[1]))
            # Add the dimension of the layer
            self.conv_dims.append(layer[1]['out_features'])

        # Setup the prediction head
        self.head = nn.Sequential(
            pooling,
            nn.Linear(self.conv_dims[-1], 1),
        )

    def forward(self, graphs):
        """
        Perform forward pass for mutagenicity prediction.

        Args:
            graphs (list of dict): 
                List of graphs, each graph is a dictionary containing the graph's attributes converted
                to torch tensors.
        Returns:
            Tensor: 
                Probability of each sample being mutagenic (n_samples, ).
        """

        # Save the results
        yhats = []
        for graph in graphs:
            # Get the graph attributes
            x, adj = graph['node_features'], graph['adj']
            # Run through the conv. layers to obtain embeddings of nodes
            for conv_layer in self.convs:
                x = conv_layer(x, adj)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Get the prediction
            yhat = self.head(x)

            # Save the result
            yhats.append(yhat)

        # Convert the result into tensor (required for loss computation)
        #  and squeeze for the correct shape (n_samples, 1) -> (n_samples, )
        yhats = torch.stack(yhats).squeeze()

        return yhats

    def __str__(self):
        """String representation of the model."""
        
        # Compute the number of layers
        num_layers = len(self.convs)

        # Compute the number of trainable parameters
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return f'GNN with {num_layers} layers and {num_params} trainable parameters.'
