"""Module implementing the various graph convolutional layers. Namely:
    - (1) Normal graph convolution
    - (2) GraphSAGE convolution (with several aggregation functions)
    - (3) Attention-based graph convolution
The following pooling layers are implemented:
    - (4) Mean pooling
    - (5) Max pooling
Finally, the following edge convolution layers are implemented:
    - (6) GraphSumEdgeConv
    - (7) GraphAttentionEdgeConv
"""

# ---------------- Import libraries and/or modules
import torch
from torch import nn
import torch.nn.functional as F
import random

# ---------------- Set random seed
torch.manual_seed(42)
random.seed(42)

# ---------------- Normal Convolution (Graph Convolution)
class GraphConv(nn.Module):
    """Simple graph convolutional layer implementing the simple neighborhood aggregation. 
    """

    name = "GraphConv"

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
    
    def __str__(self):
        """Return the name of the layer."""
        act_name = str(self.activation) if self.activation is not None else "None"
        return f"GraphConv (act: {act_name})"

# ---------------- GraphSAGE Convolution
class GraphSAGEConv(nn.Module):
    """Implementation of the GraphSAGE convolutional layer which uses a user-specified aggregation function
    to aggregate the node features of the neighborhood."""

    name = "GraphSAGEConv"
    
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
    
    def __str__(self):
        """Return the name of the layer."""
        agg_name = str(self.aggregation) if self.aggregation is not None else "None"
        act_name = str(self.activation) if self.activation is not None else "None"
        return f"GraphSAGEConv (act: {act_name}, agg: {agg_name}))"

class SumAggregation(nn.Module):
    """Aggregate node features by summing over the neighborhood."""

    name = "SumAggregation"

    def __init__(self):
        # Initialize the parent class
        super().__init__()

    def forward(self, X, adj):
        """Aggregate node features by summing over the neighborhood.

        Args:
            X (Tensor): Input node features of shape (num_nodes, in_features).
            adj (Tensor): Adjacency matrix of the graph of shape (num_nodes, num_nodes).
        Returns:
            result (Tensor): Aggregated node features of shape (num_nodes, in_features).
        """
        # Adjacency matrix serves as a mask to select the valid neighbors
        # which we then indeed sum
        result = adj @ X
        return result
    
    def __str__(self):
        """Return the name of the aggregation function."""
        return "SumAggregation"

class SqrtDegAggregation(nn.Module):
    """Aggregate node features by summing over the neighborhood and normalizing by the degrees."""

    name = "SqrtDegAggregation"

    def __init__(self):
        # Initialize the parent class
        super().__init__()

    def forward(self, X, adj):
        """Aggregate node features by summing over the neighborhood and normalizing by the degrees.

        Args:
            X (Tensor): Input node features of shape (num_nodes, in_features).
            adj (Tensor): Adjacency matrix of the graph of shape (num_nodes, num_nodes).
        Returns:
            result (Tensor): Aggregated node features of shape (num_nodes, in_features). 
        """
        # Compute the degree of each node
        # If the degree is zero, set it to one to avoid division by zero
        deg = adj.sum(1, keepdims=True).clamp(min=1)
        # Compute the sum
        nbs_sum = adj @ X

        # Normalize the sum by the sqrt degree of the node
        result = nbs_sum / deg.sqrt()

        return result
    
    def __str__(self):
        """Return the name of the aggregation function."""
        return "SqrtDegAggregation"
    
class MaxPoolAggregation(nn.Module):
    """
    Aggregate node features by taking the maximum over the transformed neighborhood.
    """

    name = "MaxPoolAggregation"

    def __init__(self):
        # Initialize the parent class
        super().__init__()

    def forward(self, X, adj):
        """
        Aggregate node features by taking the maximum over the transformed neighborhood.
        
        Args:
            X (Tensor): Input node features of shape (num_nodes, in_features).
            adj (Tensor): Adjacency matrix of the graph of shape (num_nodes, num_nodes).
        Returns:
            result (Tensor): Aggregated node features of shape (num_nodes, in_features). 
        """
        # Initialize the aggregated node embedding
        # with zeros since if the node has no neighbors
        # we want to aggregate a zero vector
        result = torch.zeros_like(X)
        n = adj.shape[0]
        for i in range(n):
            # Get 1D tensor of neighbor indices
            neighbors = adj[i].nonzero().squeeze(1)
            # Aggregate (if the node has neighbors)
            if neighbors.numel() > 0:
                result[i] = X[neighbors].max(0)[0]
        return result
    
    def __str__(self):
        """Return the name of the aggregation function."""
        return "MaxPoolAggregation"
    
# ---------------- Attention-based Convolution
class GraphAttentionConv(nn.Module):
    """Implementation of the Graph Attention convolutional layer
    which uses an attention mechanism to aggregate the node features of the neighborhood.

    Attention mechanism:
        - Compute attention scores for each neighbor
        - Apply softmax to get the attention weights
        - Compute the weighted sum of the given vector's neighbors
    """

    name = "GraphAttentionConv"

    def __init__(self, in_features, out_features, activation=None, softmax_global=False):
        # Initialize the parent class
        super().__init__()

        # Save the activation and softmax_global flag
        self.activation = activation
        self.softmax_global = softmax_global

        # Initialize the tunable parameter W and S of the layer
        # Note 1: weights are initialized with Glorot initialization
        # Note 2: S is a vector of size 2*out_features since
        # we concatenate the target node's features with the neighbor's features
        # and then we compute the dot product between S and the concat. vector
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.S = nn.Linear(2*out_features, 1, bias=False)

    def forward(self, X, adj):
        """Perform attention-based graph convolution operation.

        Args:
            X (Tensor): Input node features of shape (num_nodes, in_features).
            adj (Tensor): Adjacency matrix of the graph, shape (num_nodes, num_nodes).

        Returns:
            X_result (Tensor): Output node features after graph convolution of shape (num_nodes, out_features).
        """
        # (1) Prepare the input
        # (1a) Transform the input features
        X_prime = self.W(X)

        # (1b) Add self loops to the adjacency matrix
        # Compute the number of nodes
        n_nodes, _ = X_prime.shape

        # Set the main diagonal of adj to 0
        adj = adj - torch.diag(adj.diag())

        # Add the self loops to the adjacency matrix
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
        E_i_nb = self.S(X_prime_concat_all).squeeze()

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
        X_result = mask @ nbs_feat_weighted

        # Finally, apply activation func if applicable
        if self.activation is not None:
            X_result = torch.sigmoid(X_result)

        return X_result
    
    def __str__(self):
        """Return the name of the layer."""
        return f"GraphAttentionConv (global softmax: {self.softmax_global})"


# ---------------- Mean Pooling
class MeanPool(nn.Module):
    """Mean pooling layer."""

    name = "MeanPool"

    def __init__(self):
        # Initialize the parent class
        super().__init__()

    def forward(self, x):
        """Perform mean pooling operation."""
        return x.mean(0)
    
    def __str__(self):
        """Return the name of the layer."""
        return "MeanPool"

# ---------------- Max Pooling
class MaxPool(nn.Module):
    """Max pooling layer."""

    name = "MaxPool"

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Perform max pooling operation."""
        return x.max(0)[0]
    
    def __str__(self):
        """Return the name of the layer."""
        return "MaxPool"


# ---------------- GraphSumEdgeConv
class GraphSumEdgeConv(nn.Module):
    """Transforms the edge features and adds the sum of the neighborhood 
    edge features to the corespoinding node features.
    """

    name = "GraphSumEdgeConv"

    def __init__(self, in_features, out_features, activation=None):
        # Initialize the parent class
        super().__init__()

        # Save the activation function and local flag
        self.activation = activation

        # Linear transformation weight matrix for edges
        self.W = nn.Linear(in_features, out_features, bias=False)

    def forward(self, X, Y, edge_index):
        """Perform graph convolution operation by summing over the transformed
        edge features of the given node's neighborhood.

        Args:
            X (Tensor): Input node features of shape (num_nodes, in_features).
            Y (Tensor): Input edge features of shape (num_edges, in_features).
            edge_index (Tensor): Edge index of shape (num_edges, 2).
        Returns:
            X_prime (Tensor): Output node features after graph convolution of shape (num_nodes, num_node_out_features).
            Y_prime (Tensor): Output edge features after graph convolution of shape (num_edges, num_edge_out_features).
        """

        # Transform the edge features
        Y_prime = self.W(Y)
 
        # Sum the transformed edge features of the neighborhood
        n, m = X.shape
        node_edge_agg = torch.zeros(n, m, dtype=torch.double)
        src_nodes = edge_index[:, 0].to(int)
        node_edge_agg.index_add_(0, src_nodes, Y_prime)

        # Add the edge convolution to the node convolution
        X_prime = X + node_edge_agg

        # Apply activation if necessary
        if self.activation is not None:
            X_prime = self.activation(X_prime)
        
        return X_prime, Y_prime

    def __str__(self):
        """Return the name of the layer."""
        act_name = str(self.activation) if self.activation is not None else "None"
        return f"GraphSumEdgeConv (act: {act_name})"

class GraphAttentionEdgeConv(nn.Module):
    """Enhances node embeddings with edge features using simple attention mechanism.

    Attention mechanism:
        - Compute attention score as a dot product between given node's features and given edge features
        - Apply softmax to get the attention coefficients (weights)
        - To each node, add the weighted sum of the edge features

    """

    name = "GraphAttentionEdgeConv"

    def __init__(self, in_features, out_features, activation=None, local=False):
        # Initialize the parent class
        super().__init__()

        # Save the activation function and local flag
        self.activation = activation
        self.local = local

        # Linear transformation weight matrices for node and edge features
        self.W = nn.Linear(in_features, out_features, bias=False)

    def _create_binary_edge_association_mask(self, edge_index, num_nodes):
        """
        Create a binary mask indicating whether each node is associated with each edge.

        Args:
            edge_index (Tensor): Edge index array of shape (num_edges, 2).
            num_nodes (int): Total number of nodes.

        Returns:
            mask (Tensor): Binary mask of shape (num_nodes, num_edges).
        """

        # Get the number of edges
        num_edges = edge_index.shape[0]

        # Create an empty binary mask
        mask = torch.zeros((num_nodes, num_edges), dtype=torch.double)

        # Fill in the binary mask based on edge_index
        for edge_idx, (src_node, target_node) in enumerate(edge_index):
            mask[src_node, edge_idx] = 1.0
            mask[target_node, edge_idx] = 1.0

        return mask

    def forward(self, X, Y, edge_index):
        """Perform graph convolution operation with attention mechanism.

        Args:
            X (Tensor): Input node features of shape (num_nodes, in_node_features).
            Y (Tensor): Input edge features of shape (num_edges, in_edge_features).
            edge_index (Tensor): Edge index of shape (2, num_edges).

        Returns:
            X_prime (Tensor): Output node features after graph convolution of shape (num_nodes, out_node_features).
        """

        # Transform edge features
        Y_prime = self.W(Y)

        # Calculate attention scores using a shared weight matrix
        attention_scores = X @ Y_prime.T # Shape (num_nodes, num_edges)

        # Apply softmax to obtain attention coefficients
        attention_weights = F.softmax(attention_scores, dim=1)

        if self.local:
            # Create a binary mask indicating whether each node is associated with each edge
            mask = self._create_binary_edge_association_mask(edge_index, X.shape[0])
            # Apply the mask to the attention weights
            attention_weights = attention_weights * mask

        # Apply attention to edge features
        attentive_edge_features = attention_weights @ Y_prime # Shape (num_nodes, out_features) 

        # Add the attentive edge features to the node features
        X_prime = X + attentive_edge_features

        # Apply activation if necessary
        if self.activation is not None:
            X_prime = self.activation(X_prime)

        return X_prime, Y_prime

    def __str__(self):
        """Return the name of the layer."""
        act_name = str(self.activation) if self.activation is not None else "None"
        return f"GraphAttentionEdgeConv (act: {act_name}, local: {self.local})"
