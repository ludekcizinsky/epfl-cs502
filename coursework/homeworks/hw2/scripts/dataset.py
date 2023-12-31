"""Implementation of custom dataset class for the graph classification task.
"""

# ---------------- Import libraries and/or modules
from torch.utils.data import Dataset
import torch

# ---------------- Graph Dataset
class GraphDataset(Dataset):
    """A custom dataset for the graph classification task."""
    def __init__(self, X, y):
        # Save the data as a list of graphs represented as dictionaries
        self.dataset = [self._raw2graph(unparsed_graph) for unparsed_graph in X]
        # Save the labels
        self.y = y
        # Save the number of graphs
        self.num_graphs = len(X)
    
    @staticmethod
    def _raw2graph(unparsed_graph):
        """Parse a graph from the raw dataset.

        Args:
            unparsed_graph (datasets.arrow_dataset.Dataset): 
                A graph from the raw dataset.

        Returns:
            graph (dict):
                A dictionary containing the graph's attributes converted
                to torch tensors.
        """

        # Save result to a dictionary
        graph = dict()

        # High level info
        graph['num_nodes'] = unparsed_graph["num_nodes"]

        # Node and edge features
        # Note: dtype=torch.double is to increase precision when computing the gradient etc.
        graph['node_features'] = torch.tensor(unparsed_graph["node_feat"], dtype=torch.double)
        graph['edge_features'] = torch.tensor(unparsed_graph["edge_attr"], dtype=torch.double)

        # Adjacency matrix
        adj = torch.zeros((graph['num_nodes'], graph['num_nodes']), dtype=torch.double)
        row_indices, column_indices = unparsed_graph["edge_index"]
        adj[row_indices, column_indices] = 1
        graph['adj'] = adj

        # Edge index
        graph['edge_index'] = torch.tensor(unparsed_graph["edge_index"], dtype=torch.long).T

        return graph

    def __getitem__(self, idx):
        return self.dataset[idx], self.y[idx]
    
    def __len__(self):
        return self.num_graphs