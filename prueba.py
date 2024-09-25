import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


np.random.seed(34)
num_Datos = 1000


humedadSuelo = np.random.uniform(10, 70, num_Datos)
temperatura = np.random.uniform(25, 45, num_Datos)
RiegoSiNo = (humedadSuelo < 25) & (temperatura > 25) 
RiegoSiNo = RiegoSiNo.astype(int)

data = pd.DataFrame({
    'Humedad_Suelo': humedadSuelo,
    'Temperatura': temperatura,
    'Necesita_Riego': RiegoSiNo
})

#data.to_csv('datos_riego.csv', index=False)

X = data[['Humedad_Suelo', 'Temperatura']].values
y = data['Necesita_Riego'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class RiegoModelo(nn.Module):
    def __init__(self):
        super(RiegoModelo, self).__init__()
        self.fc1 = nn.Linear(2, 5)  # Capa de entrada
        self.fc2 = nn.Linear(5, 3)   # Capa oculta
        self.fc3 = nn.Linear(3, 1)   # Capa de salida
        self.sigmoid = nn.Sigmoid()  # Función de activación

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = RiegoModelo()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)


losses = []

num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    
    optimizer.zero_grad() 
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
model.eval()
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
    
    y_pred = model(X_test_tensor)
    y_pred_classes = (y_pred >= 0.5).float()

    accuracy = (y_pred_classes.eq(y_test_tensor).sum() / float(y_test_tensor.size(0))).item()
    print(f'Precisión del modelo: {accuracy:.4f}')

plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, num_epochs + 1), losses, label='Pérdida')
plt.title('Pérdida Durante el Entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))


X_test_unscaled = scaler.inverse_transform(X_test)  # Desescalar para graficar


plt.scatter(X_test_unscaled[:, 0], y_test, color='blue', label='Datos Reales', alpha=0.5)
plt.scatter(X_test_unscaled[:, 0], y_pred_classes.numpy(), color='red', label='Predicciones', alpha=0.5)

plt.title('Predicciones vs. Datos Reales')
plt.xlabel('Humedad del Suelo')
plt.ylabel('Necesita Riego (0 o 1)')
plt.legend()
plt.grid()
plt.show()


torch.save(model.state_dict(), 'modelo_riego.pth')
