!7z x drive/MyDrive/data.7z



import torch
import torch.nn as nn
import numpy as np
import random
import os
import copy
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f" Работаем на: {device}")



class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.subset[index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)

def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

train_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


DATA_DIR = './data/data' if os.path.exists('./data/data') else './data'

full_dataset = datasets.ImageFolder(root=DATA_DIR, loader=rgb_loader)


train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

raw_train, raw_val, raw_test = random_split(full_dataset, [train_size, val_size, test_size])


train_dataset = TransformedSubset(raw_train, transform=train_transform)
val_dataset = TransformedSubset(raw_val, transform=val_test_transform)
test_dataset = TransformedSubset(raw_test, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f" Обучение: {len(train_dataset)} | Валидация: {len(val_dataset)} | Тест: {len(test_dataset)}")
print(f" Классы в работе: {full_dataset.classes}")


from torchvision.models import resnet50, ResNet50_Weights

model = resnet50(weights=ResNet50_Weights.DEFAULT)


for param in model.parameters():
    param.requires_grad = False


num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 6)
model = model.to(device)

criterion = nn.CrossEntropyLoss()


def run_epoch(model, loader, criterion, optimizer=None, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

    return running_loss / total, (correct.double() / total).item()




history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print("ЭТАП 1: Обучение классификатора")
optimizer_head = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-2)

for epoch in range(3):
    tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer_head, is_train=True)
    val_loss, val_acc = run_epoch(model, val_loader, criterion, is_train=False)

    history['train_loss'].append(tr_loss)
    history['train_acc'].append(tr_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    print(f"Эпоха {epoch+1}/10 | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")


print("\n ЭТАП 2: дообучение всей сети")
for param in model.parameters():
    param.requires_grad = True

optimizer_full = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)

best_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(7):
    tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer_full, is_train=True)
    val_loss, val_acc = run_epoch(model, val_loader, criterion, is_train=False)

    history['train_loss'].append(tr_loss)
    history['train_acc'].append(tr_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    print(f"Эпоха {epoch+4}/10 | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())

model.load_state_dict(best_model_wts)
print(f"\n Обучение завершено. Лучшая точность: {best_acc:.4f}")


plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Обучение (Train Loss)', color='blue', lw=2)
plt.plot(history['val_loss'], label='Валидация (Val Loss)', color='orange', lw=2)
plt.axvline(x=2.5, color='red', linestyle='--', label='Начало Fine-Tuning')
plt.title('График функции потерь (Loss)')
plt.xlabel('Эпоха')
plt.ylabel('Значение Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Обучение (Train Acc)', color='blue', lw=2)
plt.plot(history['val_acc'], label='Валидация (Val Acc)', color='orange', lw=2)
plt.axvline(x=2.5, color='red', linestyle='--', label='Начало Fine-Tuning')
plt.title('График доли верных ответов (Accuracy)')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()
plt.grid(True)

plt.show()


model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(" ИТОГОВЫЕ МЕТРИКИ НА ТЕСТОВОЙ ВЫБОРКЕ:")
print("-" * 60)
print(classification_report(all_labels, all_preds, target_names=full_dataset.classes))
print("-" * 60)

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=full_dataset.classes, yticklabels=full_dataset.classes, cmap='Blues', cbar=False)
plt.title('Матрица ошибок классификации автомобилей', fontsize=14)
plt.ylabel('Истинные классы', fontsize=12)
plt.xlabel('Предсказанные классы', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np

def visualize_results(model, dataset, num_samples=6):
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    plt.figure(figsize=(15, 10))

    # обратное преобразование 
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_tensor, label = dataset[idx]
            img_gpu = img_tensor.unsqueeze(0).to(device)

            outputs = model(img_gpu)
            _, pred = torch.max(outputs, 1)

            #подготовка изображения
            img_vis = inv_normalize(img_tensor).permute(1, 2, 0).numpy()
            img_vis = np.clip(img_vis, 0, 1)

            plt.subplot(2, 3, i + 1)
            plt.imshow(img_vis)

            pred_class = full_dataset.classes[pred.item()]
            true_class = full_dataset.classes[label]

            color = 'green' if pred.item() == label else 'red'
            plt.title(f"Pred: {pred_class}\nTrue: {true_class}", color=color, fontsize=10)
            plt.axis('off')

    plt.tight_layout()
    plt.show()


visualize_results(model, test_dataset)
