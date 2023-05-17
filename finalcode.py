import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from sklearn.metrics import average_precision_score
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageFile, ImageFont, ImageDraw
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
import warnings
resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

num_epochs = 100
batch_size = 40
learning_rate = 0.01
num_classes = 10

# Chuẩn bị dữ liệu
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Tải dữ liệu từ URL
train_dataset = datasets.ImageFolder('/AI images/data100/train', transform=train_transform)
val_dataset = datasets.ImageFolder('/AI images/data100/val', transform=val_transform)

print(f"------Images in train dataset: {len(train_dataset)} / {len(train_dataset)}-----")
print(f"-----Images in val dataset: {len(val_dataset)} / {len(val_dataset)}-----")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print('Đã tải dữ liệu...')

# ======== Định nghĩa model Resnet50 ========
# Định nghĩa mô hình ResNet50
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.to(device)
print('Đã định nghĩa các model...')
print(nn.Linear(num_ftrs, num_classes))
print(num_ftrs)

# tự tạo folder tên saved_images trong folder chứa chương trình
os.makedirs("saved_images", exist_ok=True)
print('Đã tạo folder lưu ảnh kiểm tra model...')

# ========== Gắn label ================
# Truy cập vào danh sách tên lớp
class_names = val_dataset.classes
# Truy cập vào label và in ra tên lớp tương ứng
for data, target in val_loader:
    data, target = data.to(device), target.to(device)
    output = model(data)
    val_pred = output.argmax(dim=1, keepdim=True)
    for i in range(len(target)):
        label = target[i].item()
        class_name = class_names[label]
        print(f"Label: {label}, Class name: {class_name}")
print('Đã gắn label...')

# ========= Random 10 ảnh ==============
# Tạo list chứa tên các folder
folders = train_dataset.classes
# Random 10 folder
selected_folders = random.sample(folders, 10)

# ========== Định nghĩa hàm loss và optimizer ===========
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# ========== Định nghĩa hàm tính mAP ===========
def calculate_mAP(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            all_predictions.append(output)
            all_targets.append(target)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        average_precision = []
        for i in range(num_classes):
            y_true = (all_targets == i).float()
            y_scores = all_predictions[:, i]
            average_precision.append(average_precision_score(y_true.cpu(), y_scores.cpu()))
        mAP = np.mean(average_precision)
    return mAP


# ========== Huấn luyện và đánh giá mô hình ===========
train_accs = []
val_accs = []
mAPs = []
best_val_acc = 0
for epoch in range(num_epochs):
    # Huấn luyện
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    # Đánh giá trên tập huấn luyện(train acc)
    model.eval()
    train_correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_pred = output.argmax(dim=1, keepdim=True)
            train_correct += train_pred.eq(target.view_as(train_pred)).sum().item()
    train_acc = train_correct / len(train_loader.dataset)
    train_accs.append(train_acc)

    # Đánh giá trên tập validation(val acc)
    model.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            val_pred = output.argmax(dim=1, keepdim=True)
            val_correct += val_pred.eq(target.view_as(val_pred)).sum().item()
    val_loss /= len(val_loader.dataset)
    val_acc = val_correct / len(val_loader.dataset)
    val_accs.append(val_acc)

    # Tính mAP
    mAP = calculate_mAP(model, val_loader, device)
    mAPs.append(mAP)


    print('Epoch: {}   | Val Loss: {:.4f} | Train Acc: {:.4f} | Val Acc: {:.4f} | mAP: {:.4f}'.format(
        epoch + 1, val_loss, train_acc, val_acc, mAP))
    #========= Lưu lại trạng thái của model thành file pth ============
    if val_acc > best_val_acc:
        # Lưu model
        save_path = "saved_models/model_fruit_classify_wnone2.pth"
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'val_acc': val_acc
        }
        # cập nhật lại val acc tốt nhất
        torch.save(checkpoint, save_path)
        best_val_acc = val_acc
        print(f"Saved model with val acc {best_val_acc:.4f}...")
        # In ra một vài hình ảnh trong quá trình train
    model.eval()
    with torch.no_grad():
        img_count = 0
        img_list = []  # Tạo một list chứa các ảnh
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True).squeeze().cpu()
            for j in range(len(target)):
                if folders[target[j]] in selected_folders:
                    img = data[j].cpu().numpy().transpose((1, 2, 0))
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img = std * img + mean
                    img = np.clip(img, 0, 1)
                    plt.imshow(img)
                    plt.title(f"Label: {val_dataset.classes[target[j]]}\nPredicted: {val_dataset.classes[pred[j]]}")
                    plt.axis("off")
                    img_list.append(
                        (Image.fromarray((img * 255).astype(np.uint8)), target[j], pred[j]))  # Thêm ảnh vào list
                    img_count += 1
                    if img_count == 10:
                        break
            if img_count == 10:
                break
        # Tạo một bức ảnh mới và thêm các ảnh vào đó
        new_image = Image.new('RGB', (img_list[0][0].width * 5, img_list[0][0].height * 2))
        for i in range(10):
            new_image.paste(img_list[i][0], (img_list[0][0].width * (i % 5), img_list[0][0].height * (i // 5)))
            label = val_dataset.classes[img_list[i][1]]
            predicted = val_dataset.classes[img_list[i][2]]
            draw = ImageDraw.Draw(new_image)
            draw.text((img_list[0][0].width * (i % 5), img_list[0][0].height * (i // 5) + 5),
                      f"Label: {label}\nPredicted: {predicted}", (255, 255, 255),
                      font=ImageFont.truetype('arial.ttf', 14))
        # Lưu thành 1 bức ảnh lớn gồm 10 anh
        filename = f"saved_images/epoch_{epoch + 1}.jpg"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        new_image.save(filename)
