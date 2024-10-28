import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ucitavanje podataka
data = pd.read_csv('house-price.csv')

# Provjera SalePrice stupca. Ako su vrijednosti formatirane s tockama, ukloni ih
data['SalePrice'] = data['SalePrice'].str.replace('.', '', regex=False).astype(float)

# Podjela podataka na x i y 
X = np.arange(len(data)).reshape(-1, 1)
y = data['SalePrice'].values


# Podjela na train i test skup (85:15)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Skaliranje podataka
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()


# Definiranje modela
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Sigmoid(),
            nn.Linear(16, 1)  # Iza posljednjeg sloja
        )

    def forward(self, x):
        return self.model(x)


# Definiranje hiperparametara
lr = 0.001
batch_size = 16
epochs = 1000

# Inicijalizacija modela
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Treniranje modela
for epoch in range(epochs):
    model.train()
    for i in range(0, len(X_train_scaled), batch_size):
        X_batch = torch.tensor(X_train_scaled[i:i + batch_size], dtype=torch.float32)
        y_batch = torch.tensor(y_train_scaled[i:i + batch_size], dtype=torch.float32).view(-1, 1)

        optimizer.zero_grad()
        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{epochs}] | Loss: {loss.item():.4f}")


model.eval()
with torch.no_grad():
    y_pred_scaled = model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_original = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))


# Prikaz rezultata modela
plt.figure(figsize=(10, 5))
plt.plot(y_test_original, label='Stvarne vrijednosti', color='blue')
plt.plot(y_pred, label='Predviđene vrijednosti', color='orange')
plt.xlabel('Indeks')
plt.ylabel('SalePrice')
plt.title('Stvarne vs Predviđene vrijednosti')
plt.legend()
plt.show()
