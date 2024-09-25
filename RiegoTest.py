import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np  

# Definir nuevamente la clase del modelo
class RiegoModelo(nn.Module):
    def __init__(self):
        super(RiegoModelo, self).__init__()
        self.fc1 = nn.Linear(2, 5)
        self.fc2 = nn.Linear(5, 3)
        self.fc3 = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Crear una instancia del modelo
model = RiegoModelo()

model.load_state_dict(torch.load('modelo_riego50Datos.pth', weights_only=True))

model.eval()

scaler = StandardScaler()

X_new = np.array([[25, 20], [15, 40]])  
y_real = np.array([0, 1])  


X_new_scaled = scaler.fit_transform(X_new)  
X_new_tensor = torch.FloatTensor(X_new_scaled)


with torch.no_grad():  
    predicciones = model(X_new_tensor)
    predicciones_clases = (predicciones >= 0.5).float()


print(f'Predicciones: {predicciones_clases.numpy().flatten()}')
print(f'Clases reales: {y_real}')


plt.figure(figsize=(10, 6))


plt.scatter(X_new[:, 0], y_real, color='blue', label='Clases Reales', s=100, alpha=0.6)


plt.scatter(X_new[:, 0], predicciones_clases.numpy().flatten(), color='red', label='Predicciones', s=100, alpha=0.6)

plt.xlabel('Humedad del Suelo')
plt.ylabel('Necesita Riego (0 = No, 1 = Sí)')
plt.title('Predicciones vs Clases Reales')
plt.axhline(0.5, color='gray', linestyle='--')  # Línea de referencia
plt.legend()
plt.grid()
plt.show()
