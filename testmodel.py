import torchvision.models as model
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import shutil
from torchvision.models import ResNet50_Weights
import warnings
model.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
warnings.filterwarnings("ignore")
# Define the output folder
output_folder = '/AI images/output_test'
os.makedirs(output_folder, exist_ok=True)
# Chuẩn bị dữ liệu
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# Load the saved model
model = model.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
num_classes = 10  # số lượng lớp đầu ra trong mô hình được lưu trong checkpoint
model.fc = nn.Linear(model.fc.in_features, num_classes)
checkpoint = torch.load('/AI images/saved_models/model_fruit_classify.pth')
model.load_state_dict(checkpoint['model'])
model.eval()

# Load the dataset
test_dataset = datasets.ImageFolder('/AI images/input_test/test', transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Predict and save the images
class_names = test_dataset.classes
for i, (data, target) in enumerate(test_loader):
    data = data.cpu()
    output = model(data)
    pred = output.argmax(dim=1, keepdim=True)
    class_name = class_names[pred.item()]
    filename = test_dataset.samples[i][0]
    output_path = os.path.join(output_folder, class_name)
    os.makedirs(output_path, exist_ok=True)
    shutil.copy(filename, os.path.join(output_path, f"{i}.jpg"))
    print('Dang phan loai')