from torch import nn
import torch
import torch.nn.functional as F

# ---------------- Normal Convolution (Graph Convolution)
class GraphConv(nn.Module):
    """Basic graph convolutional layer implementing the simple neighborhood aggregation."""

    def __init__(self, in_features, out_features, activation=None):
        """
        Initialize the graph convolutional layer.
        
        Args:
            in_features (int): number of input node features.
            out_features (int): number of output node features.
            activation (nn.Module or callable): activation function to apply. (optional)
        """
        super().__init__()
        self.activation = activation
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.B = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, adj):
        """
        Perform graph convolution operation.

        Args:
            x (Tensor): Input node features of shape (num_nodes, in_features).
            adj (Tensor): Adjacency matrix of the graph, shape (num_nodes, num_nodes).

        Returns:
            Tensor: Output node features after graph convolution, shape (num_nodes, out_features).
        """
        # Normalise the adjacency matrix
        adj = adj / adj.sum(1, keepdims=True).clamp(min=1)        
 
        # Compute the result
        result = self.W(adj @ x) + self.B(x)
        
        # Apply activation if neccessary
        if self.activation is not None:
            result = self.activation(result)
                
        return result

# ---------------- GraphSAGE Convolution
class GraphSAGEConv(nn.Module):
    """GraphSAGE convolutional layer."""
    
    def __init__(self, in_features, out_features, aggregation, activation=None):
        """
        Initialize the GraphSAGE convolutional layer.
        
        Args:
            in_features (int): number of input node features.
            out_features (int): number of output node features.
            aggregation (nn.Module or callable): aggregation function to apply, as x_agg = aggegration(x, adj).
            activation (nn.Module or callable): activation function to apply. (optional)
        """
        super().__init__()
        self.aggregation = aggregation
        self.activation = activation
        self.W = nn.Linear(2*in_features, out_features, bias=False)

    def forward(self, x, adj):
        """
        Perform graph convolution operation.

        Args:
            x (Tensor): Input node features of shape (num_nodes, in_features).
            adj (Tensor): Adjacency matrix of the graph, typically sparse, shape (num_nodes, num_nodes).

        Returns:
            Tensor: Output node features after graph convolution, shape (num_nodes, out_features).
        """
        aggregated = torch.cat([x, self.aggregation(x, adj)], dim=1)
        return self.activation(self.W(aggregated)) if self.activation is not None else self.W(aggregated)

# ---------------- GraphSAGE Aggregation
class MeanAggregation(nn.Module):
    """Aggregate node features by averaging over the neighborhood."""
    def __init__(self):
        super().__init__()

    def forward(self, x, adj):
        x_agg = (adj @ x) / adj.sum(1, keepdims=True).clamp(min=1)
        return x_agg
    
class SumAggregation(nn.Module):
    """Aggregate node features by summing over the neighborhood."""
    def __init__(self):
        super().__init__()

    def forward(self, x, adj):
        x_agg = adj @ x
        return x_agg

class SqrtDegAggregation(nn.Module):
    """Aggregate node features by summing over the neighborhood and normalizing by the degrees."""
    def __init__(self):
        super().__init__()

    def forward(self, x, adj):
        x_agg = (adj @ x) / adj.sum(1, keepdims=True).clamp(min=1).sqrt()
        return x_agg
    
class MaxPoolAggregation(nn.Module):
    """
    Aggregate node features by taking the maximum over the transformed neighborhood.

    Note: this is complicated to implement in pure PyTorch, so we will do a naive loop.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, adj):
        # Initialize the aggregated node embedding
        # with zeros since if the node has no neighbors
        # we want to aggregate a zero vector
        x_agg = torch.zeros_like(x)
        n = adj.shape[0]
        for i in range(n):
            # Get 1D tensor of neighbors
            neighbors = adj[i].nonzero().squeeze(1)
            # Aggregate (if the node has neighbors)
            if neighbors.numel() > 0:
                x_agg[i] = x[neighbors].max(0)[0]
        return x_agg

# ---------------- Attention-based Convolution
class GraphAttentionConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features)
        self.S = nn.Parameter(torch.randn(2*out_features))

    def forward(self, X, adj):
        """
        Perform attention-based graph convolution operation.
        Args:
            X (Tensor): Input node features of shape (num_nodes, in_features).
            adj (Tensor): Adjacency matrix of the graph, shape (num_nodes, num_nodes).
        Returns:
            Tensor: Output node features after graph convolution, shape (num_nodes, out_features).
        """

        # (1) Apply linear transformation
        X_prime = self.W(X)
        n_nodes, n_features = X_prime.shape

        X_result = torch.zeros(n_nodes, n_features, dtype=torch.double)
        for i in range(n_nodes):
            # Get i-th node's neighbors
            neighbors = adj[i].nonzero().squeeze(1) # 1D binary tensor of neighbors

            # Add the self-loop
            neighbors = torch.cat([neighbors, torch.tensor([i])])

            # Get iths node's x_prime
            x_prime_i = X_prime[i]

            # Compute the attention weights with softmax
            x_prime_neighbors = X_prime[neighbors]
            x_prime_concat_all = torch.cat([x_prime_i.repeat(len(neighbors), 1), x_prime_neighbors], dim=1)
            # Faster way to compute the attention vector: e_i_nb2 = x_prime_concat_all @ self.S
            e_i_nb = torch.zeros(len(neighbors), dtype=torch.double)
            for i, x_prime_concat in enumerate(x_prime_concat_all):
                e_i_nb[i] = torch.dot(self.S, x_prime_concat)
            
            # For comparison of the two ways of computing things: print(torch.allclose(e_i_nb, e_i_nb2))
            raw_attention_scores = F.leaky_relu_(e_i_nb)

            # Apply softmax to get the attention weights
            attention_scores = F.softmax(raw_attention_scores, dim=0)

            # Compute the weighted sum of the neighbors' x_prime
            weighted_sum = torch.matmul(x_prime_neighbors.T, attention_scores)
            
            # Finally, apply sigmoid
            X_result[i] = torch.sigmoid(weighted_sum)
        
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
