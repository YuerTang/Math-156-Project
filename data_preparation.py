import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class PoseDataset(Dataset):
    def __init__(self, annotation_file, root_dir, transform=None):
        self.annotations = []
        with open(annotation_file, 'r') as f:
            for i, line in enumerate(f):
              if i < 3 or i > 10:
                continue

              parts = line.strip().split()
              img_path = parts[0]
              target = list(map(float, parts[1:]))
              self.annotations.append((img_path, torch.tensor(target, dtype=torch.float32)))

        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_rel_path, target = self.annotations[idx]
        img_path = os.path.join(self.root_dir, img_rel_path)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, target
    

dataset = PoseDataset(annotation_file='KingsCollege/dataset_train.txt', root_dir='KingsCollege')
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

for images, targets in loader:
    print(images.shape)  # [8, 3, 224, 224]
    print(targets.shape) # [8, 7]

