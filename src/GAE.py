import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GAE


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()

        # Layer 1:
        # cached only for transductive learning
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)

        # Layer 2:
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


def __train(model, optimizer, x, train_pos_edge_index):
    model.train()
    optimizer.zero_grad()

    z = model.encode(x, train_pos_edge_index)

    # Compute loss
    loss = model.recon_loss(z, train_pos_edge_index)

    loss.backward()

    # Update parameters
    optimizer.step()

    return float(loss)


def __test(model, x, train_pos_edge_index, pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


def run(data, hidden_channels, out_channels=10, epochs=300, show_progress=True):
    # set the seed
    torch.manual_seed(42)

    num_features = data.num_features

    # model
    model = GAE(GCNEncoder(num_features, hidden_channels, out_channels))

    # move to GPU (if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    x = data.x.to(device)
    train_pos_edge_index = data.train_pos_edge_index.to(device)

    # inizialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, epochs + 1):
        loss = __train(model, optimizer, x, train_pos_edge_index)

        auc, ap = __test(
            model,
            x,
            train_pos_edge_index,
            data.test_pos_edge_index,
            data.test_neg_edge_index,
        )
        if show_progress:
            print("Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}".format(epoch, auc, ap))

    return model, x, train_pos_edge_index
