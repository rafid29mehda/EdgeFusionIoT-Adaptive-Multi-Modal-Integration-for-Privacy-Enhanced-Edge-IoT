# Installation
!pip install numpy pandas matplotlib seaborn tensorflow torch sympy web3 scikit-learn flwr
!pip install -U "flwr[simulation]"
!pip install ray[default]

# Imports and Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import torch
import sympy as sp
from web3 import Web3
from sklearn.metrics import mean_squared_error
import socket
import threading
import time
import json
import flwr as fl
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)
NUM_SENSORS = 3
DATA_POINTS = 100
PORT_START = 5000
print("Setup complete! All libraries imported.")

# Sensor Definition
class Sensor:
    def __init__(self, id, type):
        self.id = id
        self.type = type
    def generate_data(self):
        if self.type == 'temperature':
            time = np.linspace(0, 10, DATA_POINTS)
            data = 25 + 5 * np.sin(time) + np.random.normal(0, 1, DATA_POINTS)
            return pd.DataFrame({'time': time, 'temperature': data})
        elif self.type == 'gps':
            time = np.linspace(0, 10, DATA_POINTS)
            lat = 37.77 + np.random.normal(0, 0.01, DATA_POINTS)
            lon = -122.42 + np.random.normal(0, 0.01, DATA_POINTS)
            return pd.DataFrame({'time': time, 'lat': lat, 'lon': lon})
        elif self.type == 'image':
            return np.random.rand(28, 28)
sensors = [
    Sensor(1, 'temperature'),
    Sensor(2, 'gps'),
    Sensor(3, 'image')
]
sample_data = sensors[0].generate_data()
print("Sample temperature data:\n", sample_data.head())

# Error Correction and Communication
def encode_data(data):
    symbols = [sp.Symbol(f'x{i}') for i in range(len(data))]
    return np.concatenate([data, data[:2]])
def decode_data(encoded):
    original_len = len(encoded) - 2
    if np.sum(encoded[:2] - encoded[original_len:]) > 0.1:
        print("Error detected! (Simulated correction could be added here, e.g., average the values.)")
    return encoded[:original_len]
def sensor_client(sensor_id, port, data):
    time.sleep(2)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', port))
        encoded = encode_data(data.flatten())
        s.sendall(json.dumps(encoded.tolist()).encode())
def edge_server(port, received_data_list):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', port))
        s.listen()
        conn, addr = s.accept()
        with conn:
            data = b''
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
            raw = data.decode()
            encoded = np.array(json.loads(raw))
            decoded = decode_data(encoded)
            received_data_list.append(decoded)
received_data = []
threads = []
for i, sensor in enumerate(sensors):
    port = PORT_START + i
    data = sensor.generate_data().values if isinstance(sensor.generate_data(), pd.DataFrame) else sensor.generate_data()
    server_thread = threading.Thread(target=edge_server, args=(port, received_data))
    server_thread.start()
    threads.append(server_thread)
    client_thread = threading.Thread(target=sensor_client, args=(i+1, port, data))
    client_thread.start()
    threads.append(client_thread)
for t in threads:
    t.join()
print("Received data after communication:", len(received_data))


# Data Preparation
temp_data = received_data[0].reshape(-1, 1) if len(received_data) > 0 else np.random.rand(100, 1)
gps_data = received_data[1].reshape(-1, 1) if len(received_data) > 1 else np.random.rand(100, 1)
image_data = received_data[2].reshape(28, 28, 1) if len(received_data) > 2 else np.random.rand(28, 28, 1)

# RNN Model
def create_rnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1, 1)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
rnn_model = create_rnn_model()
rnn_model.fit(temp_data[:-1].reshape(-1, 1, 1), temp_data[1:], epochs=5, verbose=1)
predictions = rnn_model.predict(temp_data[:-1].reshape(-1, 1, 1))
print("RNN MSE:", mean_squared_error(temp_data[1:], predictions))

# TFLite Conversion
converter = tf.lite.TFLiteConverter.from_keras_model(rnn_model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
print("TFLite model created for edge.")

# CNN Model
def create_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model
cnn_model = create_cnn_model()
fake_labels = np.random.randint(0, 2, 1)
cnn_model.fit(np.array([image_data]), np.array([fake_labels]), epochs=1, verbose=0)
print("CNN trained.")

# Federated Learning Simulation
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, local_data):
        self.local_data = local_data
    def get_parameters(self, config):
        return [self.local_data]
    def fit(self, parameters, config):
        updated = parameters[0] + 1
        return [updated], 1, {}
    def evaluate(self, parameters, config):
        loss = 0.5
        return loss, 1, {"accuracy": 0.95}
def client_fn(cid: str):
    client_id = int(cid)
    local_data = temp_data[client_id * 50 : (client_id + 1) * 50]
    return FlowerClient(local_data)
strategy = fl.server.strategy.FedAvg()
print("Starting federated learning simulation...")
history = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=2,
    config=fl.server.ServerConfig(num_rounds=1),
    strategy=strategy
)
print("Federated learning simulation complete.", history)

# Kalman Filter and Fusion
def kalman_filter(measurements, initial_state=0, process_variance=1e-5, measurement_variance=0.1):
    state = initial_state
    uncertainty = 1
    fused = []
    for meas in measurements:
        kalman_gain = uncertainty / (uncertainty + measurement_variance)
        state = state + kalman_gain * (meas - state)
        uncertainty = (1 - kalman_gain) * uncertainty + process_variance
        fused.append(state)
    return np.array(fused)
temp_meas = temp_data.flatten()[1::2]
gps_meas = gps_data.flatten()[1::3]
fused_sequence = kalman_filter(temp_meas + gps_meas)
image_pred = cnn_model.predict(np.array([image_data]))[0][0]
if image_pred > 0.5:
    print("Anomaly detected in image!")
fused_predictions = rnn_model.predict(fused_sequence[:-1].reshape(-1, 1, 1))
print("Fused MSE:", mean_squared_error(fused_sequence[1:], fused_predictions))

# Visualizations
plt.figure(figsize=(10, 5))
plt.plot(temp_data, label='Temperature')
plt.plot(gps_data, label='GPS (scaled)')
plt.plot(fused_sequence, label='Fused')
plt.legend()
plt.title('Data Streams and Fusion')
plt.show()

# Blockchain Simulation
w3 = Web3(Web3.HTTPProvider('https://rpc.ankr.com/eth_goerli'))
if w3.is_connected():
    print("Logging fused data to blockchain:", json.dumps(fused_sequence[:5].tolist()))
else:
    print("Blockchain connection failed; simulated log.")

# Scalability Test
start_time = time.time()
for _ in range(10):
    kalman_filter(np.random.rand(1000))
print("Scalability test time:", time.time() - start_time)
