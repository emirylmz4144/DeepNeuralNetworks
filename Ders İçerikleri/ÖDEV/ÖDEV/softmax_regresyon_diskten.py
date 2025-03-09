import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from d2l import torch as d2l
from IPython.display import display, clear_output

# Veri dizini
dataset_dir = r"C:\Users\kh\PycharmProjects\ysa1\fashion_mnist\FashionMNIST\acikhali2"

# Eğitim ve test klasörleri
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")

# Sınıf isimleri
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# **Özel PyTorch Dataset Sınıfı**
class FashionMNISTDataset(Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.images = []
        self.labels = []

        labels_path = os.path.join(image_folder, "labels.txt")

        with open(labels_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            image_name, label = parts[0], int(parts[1])  # Sadece ilk iki sütunu al
            image_path = os.path.join(image_folder, image_name)

            self.images.append(image_path)
            self.labels.append(label)

        print(f"✅ {len(self.images)} görüntü yüklendi!")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Görüntüyü aç ve tensöre dönüştür
        image = Image.open(self.images[idx]).convert("L")  # Grayscale
        image = np.array(image, dtype=np.float32) / 255.0  # Normalize et
        image = torch.tensor(image).unsqueeze(0)  # [1, 28, 28] şekline getir

        label = self.labels[idx]
        return image, label


# Dataset'leri oluştur
train_dataset = FashionMNISTDataset(train_dir)
test_dataset = FashionMNISTDataset(test_dir)

# DataLoader'ları tanımla
batch_size = 128
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model Parametreleri
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# **Grafik Çizme İçin Güncellenmiş Eğitim Fonksiyonu**
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    train_losses, train_accuracies, test_accuracies = [], [], []

    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)

        train_losses.append(train_metrics[0])
        train_accuracies.append(train_metrics[1])
        test_accuracies.append(test_acc)

        print(
            f"Epoch {epoch + 1}: Loss={train_metrics[0]:.4f}, Train Acc={train_metrics[1]:.4f}, Test Acc={test_acc:.4f}")

    # **Grafikleri Çizdir**
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].plot(range(1, num_epochs + 1), train_losses, label="Eğitim Kaybı (Loss)")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Eğitim Kaybı (Loss) Grafiği")
    axs[0].legend()

    axs[1].plot(range(1, num_epochs + 1), train_accuracies, label="Eğitim Doğruluğu (Train Acc)")
    axs[1].plot(range(1, num_epochs + 1), test_accuracies, label="Test Doğruluğu (Test Acc)")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_title("Eğitim ve Test Doğruluk Grafiği")
    axs[1].legend()

    plt.show()


def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


lr = 0.001


def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)#sgd algoritması uygulanıyor


num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)


# **Tahmin Fonksiyonu**
def predict_ch3(net, test_iter, n=6):
    for X, y in test_iter:
        break
    X = X[:n]
    y = y[:n]

    y_hat = net(X.reshape((-1, 784))).argmax(axis=1)

    fig, axes = plt.subplots(1, n, figsize=(10, 10))

    for i in range(n):
        img = X[i].reshape(28, 28).numpy()
        true_label = class_names[y[i].item()]
        pred_label = class_names[y_hat[i].item()]

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"{true_label}\n({pred_label})", fontsize=10,
                          color=("green" if true_label == pred_label else "red"))
        axes[i].axis('off')

    plt.show()


# **Tahminleri Gör**
predict_ch3(net, test_iter, n=6)
