pip install torch torch-geometric numpy

import torch
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d

class InteractionNetwork(MessagePassing):
    def __init__(self, node_feature_size, vertex_feature_size):
        super(InteractionNetwork, self).__init__(aggr='add')  # "Add" aggregation.
        # Interaction networks for particle-particle interactions
        self.mlp1 = Seq(Lin(2 * node_feature_size, 128), ReLU(), Lin(128, node_feature_size))
        # Interaction networks for vertex-particle interactions
        self.mlp2 = Seq(Lin(node_feature_size + vertex_feature_size, 128), ReLU(), Lin(128, node_feature_size))
        # Output layer
        self.mlp3 = Seq(Lin(node_feature_size, 64), ReLU(), Lin(64, 1), nn.Sigmoid())

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, node_feature_size]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i has shape [E, node_feature_size] for source nodes
        # x_j has shape [E, node_feature_size] for target nodes
        # edge_attr has shape [E, vertex_feature_size] for edge attributes
        tmp = torch.cat([x_i, x_j], dim=1)  # Particle-particle interaction
        tmp = self.mlp1(tmp)
        tmp += torch.cat([x_i, edge_attr], dim=1)  # Vertex-particle interaction
        return self.mlp2(tmp)

    def update(self, aggr_out):
        # aggr_out has shape [N, node_feature_size] which is the output of aggregation
        return self.mlp3(aggr_out)

# Data Preparation
# Load data into 'data_list', where each data is a PyG Data object
# Example: data = Data(x=node_features, edge_index=edge_indices, edge_attr=edge_attributes, y=labels)
# node_features: [num_particles, num_features_per_particle]
# edge_indices: [2, num_edges]
# edge_attributes: [num_edges, num_features_per_vertex]
# labels: [num_particles, 1]

data_list = [...]  # This should be filled with actual data loading mechanism
loader = DataLoader(data_list, batch_size=32, shuffle=True)

# Model Initialization
node_feature_size = 30  # Adjust based on your dataset
vertex_feature_size = 14  # Adjust based on your dataset
model = InteractionNetwork(node_feature_size, vertex_feature_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Training Loop
def train():
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Validation Loop
def validate(loader):
    model.eval()
    total_loss = 0
    correct = 0
    for data in loader:
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(out, data.y)
            total_loss += loss.item()
            pred = (out > 0.5).float()
            correct += pred.eq(data.y.view_as(pred)).sum().item()
    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader), accuracy

# Load your data into data_list
# data_list = load_your_data_function()
train_loader = DataLoader(data_list[:int(0.8 * len(data_list))], batch_size=32, shuffle=True)
val_loader = DataLoader(data_list[int(0.8 * len(data_list)):], batch_size=32, shuffle=False)

# Training and Validation
num_epochs = 100
best_val_loss = float('inf')
for epoch in range(num_epochs):
    train_loss = train()
    val_loss, val_accuracy = validate(val_loader)
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')

# Load the best model for inference or further training
# model.load_state_dict(torch.load('best_model.pth'))

# Add a testing loop if needed
def test(loader):
    model.eval()
    total_loss = 0
    correct = 0
    for data in loader:
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(out, data.y)
            total_loss += loss.item()
            pred = (out > 0.5).float()
            correct += pred.eq(data.y.view_as(pred)).sum().item()
    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader), accuracy

test_loader = DataLoader(data_list[int(0.9 * len(data_list)):], batch_size=32, shuffle=False)
test_loss, test_accuracy = test(test_loader)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')



"""
Breakdown of the Python Code
Imports and Class Definition
The code begins by importing necessary modules from PyTorch and PyTorch Geometric, which are essential for building and training neural networks on graph-structured data.
MessagePassing is the base class for creating layers that can send messages along the edges of a graph. It is a crucial component for any graph neural network.
Interaction Network Class
The InteractionNetwork class inherits from MessagePassing. It is designed to process data that represents interactions between objects (in this case, particles and vertices in high-energy physics experiments).
The network has three main components (mlp1, mlp2, mlp3), each a mini neural network (multi-layer perceptron or MLP) dealing with different aspects of the interaction data.
Constructor
__init__ defines three MLPs. mlp1 processes concatenated features from pairs of nodes (particles), mlp2 from node-vertex pairs, and mlp3 is an output layer that aggregates features to produce a prediction.
Forward Pass
The forward method orchestrates the passing of data through the network. It calls propagate, a method from MessagePassing that handles the message passing from node to node across edges defined in edge_index.
The data for each interaction type (node-node, node-vertex) is processed separately by corresponding MLPs.
Message Function
message is where the node features (x_i and x_j) and edge attributes (edge_attr) are combined and processed. This function is critical as it defines how information is transmitted across the graph.
Update Function
update is used to update node features after message passing. It aggregates the results from the message function and passes them through another MLP (mlp3) to produce final node-level outputs.
Scaling the Model
Scaling this model to handle larger datasets or to increase its complexity involves several considerations:

Hardware Acceleration
Utilize GPUs to accelerate the training and inference processes. PyTorch and PyTorch Geometric natively support GPU computation, which can significantly speed up operations on large graphs.
Data Handling
For very large graphs that don't fit into memory, consider using techniques such as graph sampling or partitioning. PyTorch Geometric supports mini-batching of graphs, which is essential for training on large datasets.
Model Complexity
Increase model capacity to handle more complex interactions by adding more layers or hidden units to the MLPs. This can help the model learn more detailed features but may also require more data and compute power.
Consider using more sophisticated types of graph neural network layers that might capture different types of interactions or long-range dependencies more effectively.
Parallel Processing
Implement data parallelism by distributing the training process across multiple GPUs. PyTorch’s DataParallel or DistributedDataParallel can be used for this purpose.
If using a cluster or cloud resources, look into PyTorch's support for distributed training which can handle training across multiple machines.
Optimization and Efficiency
Optimize the training process by tuning hyperparameters such as learning rate, batch size, and optimizer settings.
Evaluate the efficiency of message passing in your graph. If certain nodes or edges do not contribute significantly to learning, they might be pruned or ignored during training.
Evaluation and Validation
As models scale, overfitting can become a problem. Implement rigorous validation and testing procedures, possibly using a separate validation dataset to monitor for overfitting during training.
Use advanced regularization techniques (e.g., dropout, L2 regularization) tailored to graph data to improve model generalization.
Deployment
Once the model is trained and validated, consider deployment scenarios. If the model needs to make predictions in real-time, ensure that the inference pipeline is optimized for latency.
"""
