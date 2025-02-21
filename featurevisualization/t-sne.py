import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from torchvision.models import resnet50, ResNet50_Weights,resnet18, ResNet18_Weights,vgg16, VGG16_Weights, vit_b_16, ViT_B_16_Weights

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label_idx, label_name in enumerate(os.listdir(root_dir)):
            label_folder = os.path.join(root_dir, label_name)
            if os.path.isdir(label_folder):
                for file_name in os.listdir(label_folder):
                    if file_name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                        self.image_paths.append(os.path.join(label_folder, file_name))
                        self.labels.append(label_idx)
                        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset1 = ImageDataset(root_dir='SPADE_synthesized_image', transform=transform)
dataloader1 = DataLoader(dataset1, batch_size=32, shuffle=False)
dataset2 = ImageDataset(root_dir='ground_truth_class', transform=transform)
dataloader2 = DataLoader(dataset2, batch_size=32, shuffle=False)

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()
# feature_extractor = model.features
feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1])) 

def extract_features(dataloader):
    features = []
    labels = []
    with torch.no_grad():
        for images, lbls in dataloader:
            outputs = feature_extractor(images)
            outputs = outputs.view(outputs.size(0), -1)
            features.append(outputs.numpy())
            labels.extend(lbls)
    features = np.vstack(features)
    return features, labels

features1, labels1 = extract_features(dataloader1)
features2, labels2 = extract_features(dataloader2)

all_features = np.concatenate((features1, features2), axis=0)
all_labels = np.concatenate((labels1, labels2), axis=0)
dataset_labels = np.array([0]*len(features1) + [1]*len(features2))

pca = PCA(n_components=50)
features_pca = pca.fit_transform(all_features)

tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
features_tsne = tsne.fit_transform(features_pca)

plt.figure(figsize=(10, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(all_labels))))

for i, color in zip(np.unique(all_labels), colors):
    mask1 = (all_labels == i) & (dataset_labels == 0)
    mask2 = (all_labels == i) & (dataset_labels == 1)
    plt.scatter(features_tsne[mask1, 0], features_tsne[mask1, 1], color=color, marker='*', s=50)
    plt.scatter(features_tsne[mask2, 0], features_tsne[mask2, 1], color=color, marker='o', s=30)

plt.title('t-SNE visualization of image categories and datasets\nDataset 1: Star (*)  Dataset 2: Circle (o)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.savefig('/home/liaojr/BBDM-main/featurevisualization/t-SNE_Visualization_Classes_DatasetsSPADE.png')
plt.show()