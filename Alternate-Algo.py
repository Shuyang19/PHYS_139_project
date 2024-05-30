import torch
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch.utils.data import DataLoader as TorchDataLoader

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

# Ensure that the model uses GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Preparation
# Load data into 'data_list', where each data is a PyG Data object
# Example: data = Data(x=node_features, edge_index=edge_indices, edge_attr=edge_attributes, y=labels)
# node_features: [num_particles, num_features_per_particle]
# edge_indices: [2, num_edges]
# edge_attributes: [num_edges, num_features_per_vertex]
# labels: [num_particles, 1]

data_list = [...]  # This should be filled with actual data loading mechanism

# Use multi-threading for data loading
num_workers = 4  # Adjust based on your CPU
loader = TorchDataLoader(data_list, batch_size=32, shuffle=True, num_workers=num_workers)

# Model Initialization
node_feature_size = 30  # Adjust based on your dataset
vertex_feature_size = 14  # Adjust based on your dataset
model = InteractionNetwork(node_feature_size, vertex_feature_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Training Loop
def train():
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
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
        data = data.to(device)
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
train_loader = TorchDataLoader(data_list[:int(0.8 * len(data_list))], batch_size=32, shuffle=True, num_workers=num_workers)
val_loader = TorchDataLoader(data_list[int(0.8 * len(data_list)):int(0.9 * len(data_list))], batch_size=32, shuffle=False, num_workers=num_workers)
test_loader = TorchDataLoader(data_list[int(0.9 * len(data_list)):], batch_size=32, shuffle=False, num_workers=num_workers)

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
model.load_state_dict(torch.load('best_model.pth'))

# Testing Loop
def test(loader):
    model.eval()
    total_loss = 0
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(out, data.y)
            total_loss += loss.item()
            pred = (out > 0.5).float()
            correct += pred.eq(data.y.view_as(pred)).sum().item()
    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader), accuracy

test_loss, test_accuracy = test(test_loader)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
