import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Simüle Edilen Fonksiyon
def true_function(x, y):
  return x**2 + y**2

# Eğitim Verisi Oluşturma
np.random.seed(42)
num_samples = 1000
x_train = np.random.uniform(0, 10, (num_samples, 2))
y_train = true_function(x_train[:, 0], x_train[:, 1])

# Farklı Ağ Mimarileri
def get_feedforward_arch(num_neurons):
  return [
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(num_neurons, activation='relu'),
    tf.keras.layers.Dense(1)
  ]

def get_cascade_arch(num_neurons_per_layer):
  layers = [tf.keras.layers.Input(shape=(2,))]
  for num_neurons in num_neurons_per_layer:
    layers.append(tf.keras.layers.Dense(num_neurons, activation='relu'))
  layers.append(tf.keras.layers.Dense(1))
  return layers

def get_elman_arch(num_neurons_per_layer):
    layers = [tf.keras.layers.Input(shape=(2,))]  # Input shape without timestep dimension
    layers.append(tf.keras.layers.Reshape((1, 2)))  # Reshape to add timestep dimension
    layers.append(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_neurons_per_layer[0], activation='relu')))
    for num_neurons in num_neurons_per_layer[1:]:
        layers.append(tf.keras.layers.LSTM(num_neurons, return_sequences=True))
    layers.append(tf.keras.layers.LSTM(1))
    return layers


# Eğitim ve Hata Değerlendirmesi
def train_and_evaluate(model_architecture):
  # Modelin Derlenmesi
  model = tf.keras.Sequential(model_architecture)
  model.compile(optimizer='adam', loss='mean_squared_error')

  # Modelin Eğitimi
  x_train_reshaped = x_train[:, :, None]  # Reshape x_train to add a timestep dimension of size 1
  history = model.fit(x_train_reshaped, y_train, epochs=100, verbose=0)

  # Ortalama Bağıl Modelleme Hatasının Hesaplanması
  predictions = model.predict(x_train_reshaped)
  mse = np.mean((predictions - y_train)**2) / np.mean(y_train**2)
  return mse

# Farklı Konfigürasyonlar için Hataların Saklanması
errors = {}

# Farklı Ağ Türleri ve Katman Sayıları için Eğitim
errors["FF_10"] = train_and_evaluate(get_feedforward_arch(10))
errors["FF_20"] = train_and_evaluate(get_feedforward_arch(20))
errors["Cascade_20"] = train_and_evaluate(get_cascade_arch([20]))
errors["Cascade_10x2"] = train_and_evaluate(get_cascade_arch([10, 10]))
errors["Elman_15"] = train_and_evaluate(get_elman_arch([15]))
errors["Elman_3x5"] = train_and_evaluate(get_elman_arch([5, 5, 5]))

# Sonuçların Görselleştirilmesi
x_vals = np.linspace(0, 10, 100)
y_vals = np.linspace(0, 10, 100)
x_grid, y_grid = np.meshgrid(x_vals, y_vals)
input_data = np.column_stack((x_grid.flatten(), y_grid.flatten()))
x_train_reshaped = x_train[:, :, None]

plt.figure(figsize=(18, 6))

# Simüle Edilen Fonksiyon
plt.subplot(1, 3, 1)
plt.title('Simüle Edilen Fonksiyon')
plt.contourf(x_grid, y_grid, true_function(x_grid, y_grid), cmap='viridis', levels=20)
plt.colorbar()

# Elde Edilen Hata Değerlerinin Görselleştirilmesi
plt.subplot(1, 3, 2)
plt.title('Ortalama Bağıl Modelleme Hataları')
plt.bar(errors.keys(), errors.values())
plt.xticks(rotation=90)
plt.xlabel('Ağ Tipi ve Katman Sayısı')
plt.ylabel('MSE')

# Elman Ağı ile Tahminlerin Görselleştirilmesi 
elman_model = tf.keras.Sequential(get_elman_arch([15]))  # Örneğin 15 nöronlu Elman mimarisi seçildi
elman_model.compile(optimizer='adam', loss='mean_squared_error')
elman_model.fit(x_train_reshaped, y_train, epochs=100, verbose=0)
elman_predictions = elman_model.predict(input_data).reshape(x_grid.shape)

plt.subplot(1, 3, 3)
plt.title('Elman Ağı Tahmini')
plt.contourf(x_grid, y_grid, elman_predictions, cmap='viridis', levels=20)
plt.colorbar()

plt.tight_layout()
plt.show()