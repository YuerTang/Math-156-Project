

# Testing
test_dataset = PoseDataset(annotation_file='KingsCollege/dataset_test.txt', root_dir='KingsCollege', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)


model.eval()  
test_loss = 0.0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)         
        targets = targets.float().to(device)
        outputs = model(inputs)              
        loss = criterion(outputs, targets) 
        test_loss += loss.item() * inputs.size(0)  
avg_test_loss = test_loss / len(test_loader.dataset)
print(f"Test MSE: {avg_test_loss:.4f}")