import torch
import torch.nn as nn
from src.ResNetCBAM import ResNetCBAM, BasicBlockWithCBAM
train_dataset = PoseDataset(annotation_file='KingsCollege/dataset_train.txt', root_dir='KingsCollege')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

model = ResNetCBAM(BasicBlockWithCBAM, [2, 2, 2, 2], num_outputs=7)  # ResNet18-like
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(100):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        if epoch % 20 == 0:
            print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}: Loss = {running_loss:.4f}")