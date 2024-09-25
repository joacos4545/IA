import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#--------------------------------------------------

# np.random.seed(34)
# num_Datos = 1000
# humedadSuelo = np.random.uniform(10, 70, num_Datos)
# temperatura = np.random.uniform(15, 45, num_Datos)
# RiegoSiNo = (humedadSuelo < 25) & (temperatura > 25) 
# RiegoSiNo = RiegoSiNo.astype(int)
# data = pd.DataFrame({
#     'Humedad_Suelo': humedadSuelo,
#     'Temperatura': temperatura,
#     'Necesita_Riego': RiegoSiNo
# })

#--------------------------------------------------

# print(data.head())
# data.to_csv('datos_riego.csv', index=False)

#--------------------------------------------------
data = pd.read_csv('datos_riego.csv')
print(data.head())

#--------------------------------------------------

#Separaramos los datos
X = data[['Humedad_Suelo', 'Temperatura']].values
y = data['Necesita_Riego'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Creamos el modelo
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
    
# Inicializamos el modelo
model = RiegoModelo()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

#datos a tensores de PyTorch
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    
# Entrenamiento
num_epochs = 50
for epoch in range(num_epochs):
    model.train()

    optimizer.zero_grad()
    
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)  

    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluamos el modelo
model.eval()
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
    
    y_pred = model(X_test_tensor)
    y_pred_classes = (y_pred >= 0.5).float()
    
    accuracy = (y_pred_classes.eq(y_test_tensor).sum() / float(y_test_tensor.size(0))).item()
    print(f'Precisión del modelo: {accuracy:.4f}')

# Guardar el modelo
torch.save(model.state_dict(), 'modelo_riego2.pth')