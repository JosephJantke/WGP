from torch.utils.data import DataLoader, TensorDataset

#todo GPT code

# X: tensor of shape [200, 1, H, W], Y: tensor of shape [200]
dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# validation
val_set = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_set, batch_size=32)
