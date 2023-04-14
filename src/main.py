import torch
import torchvision
from PIL import Image
from load_data import train_test_set_loader

PANOS = False

# make datasets
train_test_set_loader(path="output/", test_size=0.1, val_size=0.1 ,example=True, panos=PANOS)

# Load the dataset
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.ImageFolder('train_directory/', transform=transform)
print(trainset.classes)
print(trainset.class_to_idx)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=PANOS) # if we use singles (panos=False) we should not shuffle

testset = torchvision.datasets.ImageFolder('test_directory/', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=PANOS)

valset = torchvision.datasets.ImageFolder('val_directory/', transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=PANOS)

# Load the pretrained model
model = torchvision.models.vgg16(pretrained=True)

# finetune the model
# TODO

