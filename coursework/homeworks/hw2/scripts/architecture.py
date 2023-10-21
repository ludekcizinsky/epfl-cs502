from torch import nn
from torch.nn import functional as F
import torch


class GNN(nn.Module):
    """A full Graph Neural Network for mutagenicity prediction."""

    def __init__(self, architecture, pooling, dropout=0.1):
        """
        Initialize the network.
        Args:
            architecture(list of lists): 
                List of layers, each layer is a list of the given layer type and arguments for the layer
                passed as a dictionary.
            pooling (nn.Module or callable):
                Pooling function to apply, as x_pooled = pooling(x).
            dropout (float):
                Dropout probability. (optional)
        """
        super().__init__() 

        # Hyper-parameters
        self.dropout = dropout
        
        # Deep Graph Convolutional Architecture
        self.convs = nn.ModuleList()
        self.conv_dims = []
        for layer in architecture:
            # Add the layer
            self.convs.append(layer[0](**layer[1]))
            # Add the dimension of the layer
            self.conv_dims.append(layer[1]['out_features'])

        # Add batch normalisation
        self.norms = nn.ModuleList([nn.BatchNorm1d(dim) for dim in self.conv_dims])

        # Setup the prediction head
        self.pooling = pooling
        self.head = nn.Sequential(
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
        yhats = []
        for graph in graphs:
            # Get the graph attributes
            x, adj = graph['node_features'], graph['adj']
            # Run through the conv. layers to obtain embeddings of nodes
            for conv_layer, norm_layer in zip(self.convs, self.norms):
                x = conv_layer(x, adj)
                # x = norm_layer(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Get the prediction
            x = self.pooling(x)
            yhat = self.head(x)

            # Save the result
            yhats.append(yhat)
        
        yhats = torch.stack(yhats).squeeze()

        return yhats
