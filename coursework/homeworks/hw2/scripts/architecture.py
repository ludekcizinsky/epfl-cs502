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

    def __init__(self, node_architecture, pooling, dropout=0.1, edge_architecture=None):
        """
        Initialize the network.
        Args:
            node_architecture (list):
                Type of the layer, dimensions, hyperparameters for the layer.
            edge_architecture (list):
                Type of the layer, dimensions, hyperparameters for the layer.
            pooling (nn.Module or callable):
                Pooling function to apply, as x_pooled = pooling(x).
            dropout (float):
                Dropout probability. (optional)
        """
        # Initialize the parent class
        super().__init__() 

        # Save the dropout probability
        self.dropout = dropout
        self.use_edges = edge_architecture is not None

        # Parse node architecture info
        node_conv, node_dims, node_conv_kwargs = node_architecture

        # Parse edge integration info
        if self.use_edges:
            edge_conv, edge_dims, edge_conv_kwargs = edge_architecture
        
        # Deep Graph Convolutional Architecture
        self.convs = nn.ModuleList()
        for out_dim_index in range(1, len(node_dims)):
            # Get the in/out dimension
            in_dim = node_dims[out_dim_index - 1]
            out_dim = node_dims[out_dim_index]

            # Add the node convolutional layer
            l = node_conv(in_dim, out_dim, **node_conv_kwargs)
            self.convs.append(l)

            # Add the edge convolutional layer (optional)
            if self.use_edges:
                # Get the in/out dimension
                in_dim = edge_dims[out_dim_index - 1]
                out_dim = edge_dims[out_dim_index]

                # Add the edge convolutional layer
                l = edge_conv(in_dim, out_dim, **edge_conv_kwargs)
                self.convs.append(l)

        # Setup the prediction head
        self.head = nn.Sequential(
            pooling,
            nn.Linear(node_dims[-1], 1),
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
            # Get the node attributes
            X, adj = graph['node_features'], graph['adj']
            # get the edge attributes
            Y, edge_index = graph['edge_features'], graph['edge_index']
            # Run through the conv. layers to obtain embeddings of nodes
            i = 0
            while i < len(self.convs):
                # Define convolutional layer
                conv_layer = self.convs[i]

                # Pass through the node layer
                X = conv_layer(X, adj)

                # Pass through the edge layer (optional)
                if self.use_edges:
                    i += 1
                    conv_layer = self.convs[i]
                    X, Y = conv_layer(X, Y, edge_index)
                
                # Apply dropout (optional)
                if self.dropout > 0:
                    X = F.dropout(X, p=self.dropout, training=self.training)

                i += 1

            # Get the prediction
            yhat = self.head(X)

            # Save the result
            yhats.append(yhat)

        # Convert the result into tensor (required for loss computation)
        #  and squeeze for the correct shape (n_samples, 1) -> (n_samples, )
        yhats = torch.stack(yhats).squeeze()

        return yhats

    def _predict_nd(self, Xb, adjb):
        """Get the prediction for the given batch of graphs which
        are of the same type! Uses edges only.
        
        This function is used to determine the attribution scores of the input using the Captum library.
        
        NB: I am aware that ideally i should have a single function for both cases, but given the Captum library
        API, this is the easiest way to do it.
    
        Args:
            Xb (torch.Tensor): node features.
            adjb (torch.Tensor): adjacency matrix.
        Returns:
            yhat (torch.Tensor): prediction. 
        """
        N = Xb.shape[0]
        yhats = []
        for gi in range(N):

            # Get the ith input from the batch
            X, adj = Xb[gi], adjb[gi]
            i = 0
            while i < len(self.convs):
                # Define convolutional layer
                conv_layer = self.convs[i]

                # Pass through the node layer
                X = conv_layer(X, adj)

                # Apply dropout (optional)
                if self.dropout > 0:
                    X = F.dropout(X, p=self.dropout, training=self.training)

                i += 1

            # Get the prediction
            yhat = self.head(X)
            yhats.append(yhat)
        
        # Convert the result into tensor (required for loss computation)
        #  and squeeze for the correct shape (n_samples, 1) -> (n_samples, )
        result = torch.stack(yhats).squeeze()

        return result
 
    def _predict_nded(self, Xb, Yb, adjb, eindxb):
        """Get the prediction for the given batch of graphs which
        are of the same type! Uses both nodes and edges as features.
        
        This function is used to determine the attribution scores of the input using the Captum library.
 
        Args:
            Xb (torch.Tensor): node features.
            Yb (torch.Tensor): edge features.
            adjb (torch.Tensor): adjacency matrix.
            eindxb (torch.Tensor): edge index.
        Returns:
            yhat (torch.Tensor): prediction. 
        """
        N = Xb.shape[0]
        yhats = []
        for gi in range(N):

            # Get the ith input from the batch
            X, adj, Y, eindx = Xb[gi], adjb[gi], Yb[gi], eindxb[gi]

            i = 0
            while i < len(self.convs):
                # Define convolutional layer
                conv_layer = self.convs[i]

                # Pass through the node layer
                X = conv_layer(X, adj)

                # Pass through the edge layer
                i += 1
                conv_layer = self.convs[i]
                X, Y = conv_layer(X, Y, eindx)

                # Apply dropout (optional)
                if self.dropout > 0:
                    X = F.dropout(X, p=self.dropout, training=self.training)

                i += 1

            # Get the prediction
            yhat = self.head(X)
            yhats.append(yhat)
        
        # Convert the result into tensor (required for loss computation)
        #  and squeeze for the correct shape (n_samples, 1) -> (n_samples, )
        result = torch.stack(yhats).squeeze()

        return result

    def __str__(self):
        """String representation of the model."""
        
        # Compute the number of layers
        num_layers = len(self.convs)

        # Compute the number of trainable parameters
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return f'GNN with {num_layers} layers and {num_params} trainable parameters.'
