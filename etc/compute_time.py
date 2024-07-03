import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import time

# GPU가 사용 가능한지 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 데이터셋 로드 및 전처리
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)

# 간단한 CNN 모델 생성 함수
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 14 * 14, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 14 * 14)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델 학습 함수
def train_model(model, data_loader, criterion, optimizer, device):
    model.train()
    for epoch in range(2):
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# NumPy 배열을 사용한 모델 학습 시간 측정 (CPU)
model = SimpleCNN().to('cpu')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# NumPy로 데이터를 변환
x_train_numpy = np.array(train_dataset.data.unsqueeze(1), dtype=np.float32) / 255.0
y_train_numpy = np.array(train_dataset.targets, dtype=np.int64)
numpy_dataset = TensorDataset(torch.tensor(x_train_numpy), torch.tensor(y_train_numpy))
numpy_loader = DataLoader(dataset=numpy_dataset, batch_size=4, shuffle=True)

start_time = time.time()
train_model(model, numpy_loader, criterion, optimizer, 'cpu')
numpy_cpu_time = time.time() - start_time

# PyTorch 텐서를 사용한 모델 학습 시간 측정 (CPU)
model = SimpleCNN().to('cpu')
start_time = time.time()
train_model(model, train_loader, criterion, optimizer, 'cpu')
tensor_cpu_time = time.time() - start_time

# NumPy 배열을 사용한 모델 학습 시간 측정 (GPU, if available)
if torch.cuda.is_available():
    model = SimpleCNN().to(device)
    start_time = time.time()
    train_model(model, numpy_loader, criterion, optimizer, device)
    numpy_gpu_time = time.time() - start_time
else:
    numpy_gpu_time = None

# PyTorch 텐서를 사용한 모델 학습 시간 측정 (GPU, if available)
if torch.cuda.is_available():
    model = SimpleCNN().to(device)
    start_time = time.time()
    train_model(model, train_loader, criterion, optimizer, device)
    tensor_gpu_time = time.time() - start_time
else:
    tensor_gpu_time = None

# 결과 출력
print(f"NumPy CPU Time: {numpy_cpu_time:.2f} seconds")
print(f"PyTorch Tensor CPU Time: {tensor_cpu_time:.2f} seconds")
if torch.cuda.is_available():
    print(f"NumPy GPU Time: {numpy_gpu_time:.2f} seconds")
    print(f"PyTorch Tensor GPU Time: {tensor_gpu_time:.2f} seconds")
else:
    print("GPU not available")
