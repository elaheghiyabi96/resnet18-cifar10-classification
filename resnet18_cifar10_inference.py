#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
model = torch.load(r"C:\Users\Shayan System\Downloads\resnet18_complete_model.pth", map_location=torch.device('cpu'))
model.eval()


# In[6]:


import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Data transformations (resize + normalize)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

# Load test dataset
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

# Create test dataloader
test_loader = torch.utils.data.DataLoader(testset, batch_size=8,
                                          shuffle=False, num_workers=2)

# Function to unnormalize and display images
def imshow(img):
    img = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

# Load your saved model from specific path (update this path if needed)
model_path = r"C:\Users\Shayan System\Downloads\resnet18_complete_model.pth"
model = torch.load(model_path, map_location=device)
model.eval()
model.to(device)

# Get some test images
dataiter = iter(test_loader)
images, labels = next(dataiter)

images = images.to(device)
labels = labels.to(device)

# Predict on test images
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

# Show images with true and predicted labels
fig = plt.figure(figsize=(12,6))
for idx in range(len(images)):
    ax = fig.add_subplot(2, 4, idx+1, xticks=[], yticks=[])
    imshow(images[idx].cpu())
    ax.set_title(f'True: {classes[labels[idx]]}\nPred: {classes[predicted[idx]]}')

plt.show()


# In[ ]:




