from torch.utils.data import Dataset
import torch

class GraphDataset(Dataset):
    def __init__(self, X, y):
        self.dataset = [self._raw2graph(unparsed_graph) for unparsed_graph in X]
        self.y = y
        self.num_graphs = len(X)
    
    @staticmethod
    def _raw2graph(unparsed_graph):
        """Parse a graph from the raw dataset.

        Attributes
            unparsed_graph: datasets.arrow_dataset.Dataset
                A graph from the raw dataset.

        Returns
            graph: dict
                A dictionary containing the graph's attributes converted
                to torch tensors.
        """

        # Save result to a dictionary
        graph = dict()

        # High level info
        graph['num_nodes'] = unparsed_graph["num_nodes"]

        # Node and edge features
        graph['node_features'] = torch.tensor(unparsed_graph["node_feat"])
        graph['edge_features'] = torch.tensor(unparsed_graph["edge_attr"])

        # Adjacency matrix
        adj = torch.zeros((graph['num_nodes'], graph['num_nodes']))
        row_indices, column_indices = unparsed_graph["edge_index"]
        adj[row_indices, column_indices] = 1
        graph['adj'] = adj

        return graph

    def __getitem__(self, idx):
        return self.dataset[idx], self.y[idx]
    
    def __len__(self):
        return self.num_graphs