#todo GPT code

#set up training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SeedModel(cnn_backbone).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-4
)


#training loop
for epoch in range(15):  # use early stopping if needed
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device).float()
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

    # Validation step
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            preds = model(xb)
            val_loss += criterion(preds, yb).item()
            predicted = (preds > 0.5).int()
            correct += (predicted == yb.int()).sum().item()
            total += yb.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Accuracy: {acc:.2%}")