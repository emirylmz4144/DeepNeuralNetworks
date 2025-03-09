import torch
import torch.nn as nn
import torch.optim as optim

# **1. Giriş ve Çıkış Verileri (Mantıksal AND Kapısı)**
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)  # Girişler
y = torch.tensor([0, 0, 0, 1], dtype=torch.long)  # Gerçek etiketler

# **2. Softmax Regresyon Modeli**
class SoftmaxAND(nn.Module):  #Giriş olarak (A, B) değerlerini alır. İki çıkış üretir (Sınıf 0 ve Sınıf 1 için skorlar).
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)  # 2 giriş, 2 çıkış (0 ve 1 sınıfları için)

    def forward(self, x):
        return self.linear(x)  # Softmax aktivasyonu uygula (Yani Skorları olasılıklara dönüştürür ve
                                                    # en yüksek olasılığa sahip sınıf tahmin edilir.

# **3. Modeli Tanımla**
model = SoftmaxAND() #Model, girişleri alıp olasılık tahmini yapmaya hazır hale gelir.

# **4. Kayıp Fonksiyonu ve Optimizasyon**
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss (Softmax ile kullanılır)
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Stokastik Gradient Descent

# **5. Modeli Eğitme**
epochs = 100  # 5000 iterasyon
for epoch in range(epochs):
    optimizer.zero_grad()  # Gradyanları sıfırla
    outputs = model(X)  # Modelden çıktı al
    loss = criterion(outputs, y)  # Gerçek değerlerle karşılaştır ve kaybı hesapla.
    loss.backward()  # Geri yayılım (backpropagation) yani Backpropagation yaparak gradyanları hesaplar
    optimizer.step()  # Parametreleri yani  Ağırlıkları günceller

    if epoch % 500 == 0:  # Her 500 iterasyonda bir ekrana yazdır
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# **6. Modelin Tahminleri**
print("\n==> Eğitim Tamamlandı. Modelin Tahminleri:")
with torch.no_grad():
    predictions = model(X).argmax(dim=1)  # En yüksek olasılığa sahip sınıfı seç
    for i in range(len(X)):
        print(f"Giriş: {X[i].tolist()} => Tahmin: {predictions[i].item()} (Gerçek: {y[i].item()})")

