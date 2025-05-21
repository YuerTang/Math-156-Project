import torch
import torch.nn as nn
# from ..data_preparation import PoseDataset

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 2 conv + maxpool
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 2 conv + maxpool
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 3 conv + maxpool
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 3 conv + maxpool
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5: 3 conv + maxpool
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), 
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096, 7)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x

# from torchvision import transforms

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),    
#     transforms.ToTensor(),         
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],   
#         std=[0.229, 0.224, 0.225]
#     )
# ])

# # Training
# train_dataset = PoseDataset(annotation_file='KingsCollege/dataset_train.txt', root_dir='KingsCollege', transform=transform)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# model = VGG16()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# for epoch in range(10):
#     model.train()
#     running_loss = 0.0
#     for imgs, labels in train_loader:
#         imgs, labels = imgs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(imgs)

#         loss = criterion(outputs.squeeze(), labels.float())
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
    
#     print(f"Epoch {epoch+1}: Loss = {running_loss:.4f}")

# # Testing
# test_dataset = PoseDataset(annotation_file='KingsCollege/dataset_test.txt', root_dir='KingsCollege', transform=transform)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)


# model.eval()  
# test_loss = 0.0

# with torch.no_grad():
#     for inputs, targets in test_loader:
#         inputs = inputs.to(device)         
#         targets = targets.float().to(device)
#         outputs = model(inputs)              
#         loss = criterion(outputs, targets) 
#         test_loss += loss.item() * inputs.size(0)  
# avg_test_loss = test_loss / len(test_loader.dataset)
# print(f"Test MSE: {avg_test_loss:.4f}")

