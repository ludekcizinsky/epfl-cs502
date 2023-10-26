"""Module implementing the various graph convolutional layers. Namely:
    - (1) Normal graph convolution
    - (2) GraphSAGE convolution (with Mean aggregation)
    - (3) Attention-based graph convolution
Finally, the following pooling layers are implemented:
    - (4) Mean pooling
    - (5) Max pooling
"""

# ---------------- Import libraries and/or modules
import torch
from torch import nn
import torch.nn.functional as F

# ---------------- Normal Convolution (Graph Convolution)
class GraphConv(nn.Module):
    """Simple graph convolutional layer implementing the simple neighborhood aggregation. 
    """

    def __init__(self, in_features, out_features, activation=None):
        """
        Initialize the graph convolutional layer.
        
        Args:
            in_features (int): number of input node features.
            out_features (int): number of output node features.
            activation (nn.Module or callable): activation function to apply. (optional)
        """
        # Initialize the parent class
        super().__init__()

        # Save the activation function
        self.activation = activation

        # Initialize the tunable parameters W and B of the layer
        # Note: weights are initialized with Glorot initialization
        # to prevent the gradient from vanishing or exploding
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.B = nn.Linear(in_features, out_features, bias=False)

    def forward(self, X, adj):
        """
        Perform graph convolution operation.

        Args:
            X (Tensor): Input node features of shape (num_nodes, in_features).
            adj (Tensor): Adjacency matrix of the graph of shape (num_nodes, num_nodes).

        Returns:
            result (Tensor): Output node features after graph convolution of shape (num_nodes, out_features).
        """
        # Normalise the adjacency matrix
        # Note: we use clamp to avoid division by zero
        adj = adj / adj.sum(1, keepdims=True).clamp(min=1)        
 
        # Compute the result
        result = self.W(adj @ X) + self.B(X)
        
        # Apply activation if neccessary
        if self.activation is not None:
            result = self.activation(result)
                
        return result

# ---------------- GraphSAGE Convolution
class GraphSAGEConv(nn.Module):
    """Implementation of the GraphSAGE convolutional layer which uses a user-specified aggregation function
    to aggregate the node features of the neighborhood."""
    
    def __init__(self, in_features, out_features, aggregation, activation=None):
        """
        Initialize the GraphSAGE convolutional layer.
        
        Args:
            in_features (int): number of input node features.
            out_features (int): number of output node features.
            aggregation (nn.Module or callable): aggregation function to apply, ex. x_agg = aggegration(x, adj).
            activation (nn.Module or callable): activation function to apply. (optional)
        """
        # Initialize the parent class
        super().__init__()

        # Save the aggregation and activation functions
        self.aggregation = aggregation
        self.activation = activation
        
        # Initialize the tunable parameter W of the layer
        # Note: weights are initialized with Glorot initialization
        # to prevent the gradient from vanishing or exploding
        self.W = nn.Linear(2*in_features, out_features, bias=False)

    def forward(self, X, adj):
        """
        Perform graph convolution operation.

        Args:
            X (Tensor): Input node features of shape (num_nodes, in_features).
            adj (Tensor): Adjacency matrix of the graph, typically sparse of shape (num_nodes, num_nodes).

        Returns:
            result (Tensor): Output node features after graph convolution of shape (num_nodes, out_features).
        """
        # Aggregate the node features of the neighborhood
        nb_agg = self.aggregation(X, adj)

        # Concatenate the aggregated neighborhood node features 
        # with the target node features
        X_with_nb_agg = torch.cat([X, nb_agg], dim=1)

        # Transform the concatenated features using linear transformation
        result = self.W(X_with_nb_agg)

        # Apply activation if neccessary
        if self.activation is not None:
            result = self.activation(result)

        return result

class MeanAggregation(nn.Module):
    """Aggregate node features by averaging over the neighborhood."""
    def __init__(self):
        super().__init__()

    def forward(self, X, adj):
        """
        Aggregate node features by averaging over the neighborhood.
        Args:
            X (Tensor): Input node features of shape (num_nodes, in_features).
            adj (Tensor): Adjacency matrix of the graph of shape (num_nodes, num_nodes).
        Returns:
            result (Tensor): Aggregated node features of shape (num_nodes, in_features).
        """
        # Compute the degree of each node
        # If the degree is zero, set it to one to avoid division by zero
        deg = adj.sum(1, keepdims=True).clamp(min=1)

        # Normalize the adjacency matrix using the degree
        # E.g. if adj[i, j] = 1, then adj[i, j] = 1 / deg[i]
        adj_norm = adj / deg

        # Finally, aggregate the node features by averaging over the neighborhood
        result = adj_norm @ X

        return result
    
# ---------------- Attention-based Convolution
class GraphAttentionConv(nn.Module):
    """Implementation of the Graph Attention convolutional layer
    which uses an attention mechanism to aggregate the node features of the neighborhood.

    Attention mechanism:
        - Compute attention scores for each neighbor
        - Apply softmax to get the attention weights
        - Compute the weighted sum of the given vector's neighbors
    """
    def __init__(self, in_features, out_features, softmax_global=False):
        # Initialize the parent class
        super().__init__()

        # Save the softmax_global flag
        self.softmax_global = softmax_global

        # Initialize the tunable parameter W and S of the layer
        # Note 1: weights are initialized with Glorot initialization
        # Note 2: S is a vector of size 2*out_features since
        # we concatenate the target node's features with the neighbor's features
        # and then we compute the dot product between S and the concat. vector
        self.W = nn.Linear(in_features, out_features)
        self.S = nn.Parameter(torch.randn(2*out_features, dtype=torch.double)).unsqueeze(1) # Make it column vector

    def forward(self, X, adj):
        """Perform attention-based graph convolution operation.

        Args:
            X (Tensor): Input node features of shape (num_nodes, in_features).
            adj (Tensor): Adjacency matrix of the graph, shape (num_nodes, num_nodes).
        """
        # (1) Apply linear transformation on the input node features
        X_prime = self.W(X)

        # Compute the number of nodes
        n_nodes, _ = X_prime.shape

        # Add self loop to adjaceny matrix
        adj = adj + torch.eye(n_nodes, dtype=torch.double)

        # (2) Compute attention scores for each neighbor
        # (2a) Get all edges
        # E.g. if adj[i, j] = 1, then nbs will have an entry (i, j)
        # as well as (j, i) since the graph is undirected
        nbs = adj.nonzero() # of shape (2*num_edges, 2)
        i, j = nbs[:, 0], nbs[:, 1]

        # (2b) Concatenate each vector's v transformed features
        # with its neightbors' transformed features
        # E.g. if X_prime[i] = [1, 2] and X_prime[j] = [3, 4]
        # then x_prime_concat_all will include [[1, 2, 3, 4], [3, 4, 1, 2]]
        X_prime_i = X_prime[i] # Target vectors (vs)
        X_prime_j = X_prime[j] # Neighbor vectors (vnb)
        X_prime_concat_all = torch.cat([X_prime_i, X_prime_j], dim=1) # Shape: (2*num_edges, 2*out_features)

        # (2c) Compute the dot product between S and the concat. vector
        # of target vector i and its neighbor j (nb)
        # Note: (2*num_edges, 2*out_features) @ (2*out_features, 1) = (2*num_edges, 1)
        # --> squeeze result to get a 1D tensor of shape (2*num_edges, )
        E_i_nb = (X_prime_concat_all @ self.S).squeeze()

        # Apply leaky relu to get the raw attention scores
        raw_attention_scores = F.leaky_relu(E_i_nb)


        # (2d) Finally, apply the softmax function to get the attention weights
        if self.softmax_global:
            # Apply softmax globally
            attention_scores = F.softmax(raw_attention_scores, dim=0)
        else:
            # Map the values to positive range using exponential function
            exp_attention_scores = torch.exp(raw_attention_scores)

            # Create an array to hold the sum of exp. attention scores for each node
            # (i.e. the total attention of the node's neighborhood)
            neighborhood_sum = torch.zeros(n_nodes, dtype=torch.double)

            # Sum the exp. attention scores for each node's neighborhood
            # Note:
            # 0 indicates dimnesion, here we just have one dimension so 0
            # i indicates the index of the node from edge (i, j)
            neighborhood_sum.index_add_(0, i, exp_attention_scores)

            # Divide the exponential scores by the sum of the neighborhood
            attention_scores = exp_attention_scores / neighborhood_sum[i]

        # (3) Compute the weighted sum of the neighbors'
        # Create summation mask of shape (# of nodes, 2*# of edges)
        # In simple terms, the mask indicates which edges is the given node i
        # associated with. E.g. if mask[i, j] = 1, then edge (i, j) is associated.
        mask = (i.view(-1, 1) == torch.arange(n_nodes)).T.to(torch.double)

        # Weight the neighbors' features by the attention scores
        nbs_feat_weighted = attention_scores.view(-1, 1) * X_prime_j

        # Compute the weighted sum of the neighbors' x_prime
        weighted_sum = mask @ nbs_feat_weighted

        # Finally, apply sigmoid
        X_result = torch.sigmoid(weighted_sum)

        return X_result


# ---------------- Mean Pooling
class MeanPool(nn.Module):
    """Mean pooling layer."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(0)

# ---------------- Max Pooling
class MaxPool(nn.Module):
    """Max pooling layer."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.max(0)[0]
