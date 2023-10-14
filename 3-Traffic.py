import tensorflow as tf
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import os
import typing
import matplotlib.pyplot as plt
import osmnx as ox

import networkx as nx
import geopandas as gpd

# Create a road network graph
road_network = ox.graph_from_bbox(
    35.7209, 35.7042, 51.4215, 51.3856, network_type='drive')
ox.plot_graph(road_network)  # Visualize the road network

# Extract edge information from the graph
edges_df = ox.graph_to_gdfs(road_network, nodes=False, edges=True)
highways = edges_df['highway']
print("Coordinate system:", edges_df.crs)

nodes_df, edges_df = ox.graph_to_gdfs(road_network, nodes=True, edges=True)

# Compute basic statistics of the road network
network_stats = ox.basic_stats(road_network)

# Create a convex hull of the road network
edges_df.unary_union.convex_hull

# Download and extract traffic dataset
traffic_data_url = "https://github.com/VeritasYin/STGCN_IJCAI-18/raw/master/dataset/PeMSD7_Full.zip"
data_dir = tf.keras.utils.get_file(
    origin=traffic_data_url, extract=True, archive_format="zip")
data_dir = data_dir.rstrip("PeMSD7_Full.zip")

# Load traffic data
route_distances = pd.read_csv(os.path.join(
    data_dir, "PeMSD7_W_228.csv"), header=None).to_numpy()
speeds_array = pd.read_csv(os.path.join(
    data_dir, "PeMSD7_V_228.csv"), header=None).to_numpy()

edges_df['edge_index'] = np.arange(len(edges_df))

# Define sample routes and select them
sample_route_indices = edges_df['edge_index'].tolist()[:8]
selected_route_distances = route_distances[np.ix_(
    sample_route_indices, sample_route_indices)]
selected_speeds_array = speeds_array[:, sample_route_indices]

print(f"Selected route distances shape: {selected_route_distances.shape}")
print(f"Selected speeds array shape: {selected_speeds_array.shape}")

# Visualize selected data
plt.figure(figsize=(18, 6))
plt.plot(selected_speeds_array[:, [0, -1]])
plt.legend(["Route 0", "Route 25"])

# Data preprocessing parameters
train_size, val_size = 0.5, 0.2

# Data preprocessing function


def preprocess_traffic_data(data_array: np.ndarray, train_size: float, val_size: float):
    num_time_steps = data_array.shape[0]
    num_train, num_val = int(
        num_time_steps * train_size), int(num_time_steps * val_size)
    train_data = data_array[:num_train]
    mean, std = train_data.mean(axis=0), train_data.std(axis=0)
    train_data = (train_data - mean) / std
    val_data = (data_array[num_train: (num_train + num_val)] - mean) / std
    test_data = (data_array[(num_train + num_val):] - mean) / std
    return train_data, val_data, test_data


# Preprocess the traffic data
train_data, val_data, test_data = preprocess_traffic_data(
    selected_speeds_array, train_size, val_size)

print(f"Train data size: {train_data.shape}")
print(f"Validation data size: {val_data.shape}")
print(f"Test data size: {test_data.shape}")

# Create TensorFlow datasets
batch_size = 64
input_sequence_length = 12
forecast_horizon = 3
multi_horizon = False


def create_traffic_tf_dataset(data_array: np.ndarray, input_sequence_length: int, forecast_horizon: int, batch_size: int = 128, shuffle=True, multi_horizon=True):
    inputs = timeseries_dataset_from_array(
        np.expand_dims(data_array[:-forecast_horizon], axis=-1),
        None,
        sequence_length=input_sequence_length,
        shuffle=False,
        batch_size=batch_size,
    )
    target_offset = (
        input_sequence_length if multi_horizon else input_sequence_length + forecast_horizon - 1
    )
    target_seq_length = forecast_horizon if multi_horizon else 1
    targets = timeseries_dataset_from_array(
        data_array[target_offset:],
        None,
        sequence_length=target_seq_length,
        shuffle=False,
        batch_size=batch_size,
    )
    dataset = tf.data.Dataset.zip((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(100)
    return dataset.prefetch(16).cache()


train_dataset, val_dataset = (
    create_traffic_tf_dataset(data_array, input_sequence_length,
                              forecast_horizon, batch_size)
    for data_array in [train_data, val_data]
)

test_dataset = create_traffic_tf_dataset(
    test_data,
    input_sequence_length,
    forecast_horizon,
    batch_size=test_data.shape[0],
    shuffle=False,
    multi_horizon=multi_horizon,
)

# Define functions for graph processing


def compute_traffic_adjacency_matrix(
    distances_matrix: np.ndarray, sigma2: float, epsilon: float
):
    num_routes = distances_matrix.shape[0]
    distances_matrix = distances_matrix / 10000.0
    w2, w_mask = (
        distances_matrix * distances_matrix,
        np.ones([num_routes, num_routes]) - np.identity(num_routes),
    )
    return (np.exp(-w2 / sigma2) >= epsilon) * w_mask


class TrafficGraphInfo:
    def __init__(self, edges: typing.Tuple[list, list], num_nodes: int):
        self.edges = edges
        self.num_nodes = num_nodes


sigma2 = 0.1
epsilon = 0.5
traffic_adjacency_matrix = compute_traffic_adjacency_matrix(
    selected_route_distances, sigma2, epsilon)
node_indices, neighbor_indices = np.where(traffic_adjacency_matrix == 1)
graph_info = TrafficGraphInfo(
    edges=(node_indices.tolist(), neighbor_indices.tolist()),
    num_nodes=traffic_adjacency_matrix.shape[0],
)
print(
    f"Number of nodes: {graph_info.num_nodes}, Number of edges: {len(graph_info.edges[0])}")

# Define a custom GraphConvolution layer


class TrafficGraphConvolution(layers.Layer):
    def __init__(
        self,
        in_features,
        out_features,
        graph_info: TrafficGraphInfo,
        aggregation_type="mean",
        combination_type="concat",
        activation: typing.Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.graph_info = graph_info
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.weight = tf.Variable(
            initial_value=keras.initializers.glorot_uniform()(
                shape=(in_features, out_features), dtype="float32"
            ),
            trainable=True,
        )
        self.activation = layers.Activation(activation)

    def aggregate(self, neighbor_representations: tf.Tensor):
        aggregation_func = {
            "sum": tf.math.unsorted_segment_sum,
            "mean": tf.math.unsorted_segment_mean,
            "max": tf.math.unsorted_segment_max,
        }.get(self.aggregation_type)
        if aggregation_func:
            return aggregation_func(
                neighbor_representations,
                self.graph_info.edges[0],
                num_segments=self.graph_info.num_nodes,
            )
        raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")

    def compute_nodes_representation(self, features: tf.Tensor):
        return tf.matmul(features, self.weight)

    def compute_aggregated_messages(self, features: tf.Tensor):
        neighbor_representations = tf.gather(
            features, self.graph_info.edges[1])
        aggregated_messages = self.aggregate(neighbor_representations)
        return tf.matmul(aggregated_messages, self.weight)

    def update(self, nodes_representation: tf.Tensor, aggregated_messages: tf.Tensor):
        if self.combination_type == "concat":
            h = tf.concat([nodes_representation, aggregated_messages], axis=-1)
        elif self.combination_type == "add":
            h = nodes_representation + aggregated_messages
        else:
            raise ValueError(
                f"Invalid combination type: {self.combination_type}.")
        return self.activation(h)

    def call(self, features: tf.Tensor):
        nodes_representation = self.compute_nodes_representation(features)
        aggregated_messages = self.compute_aggregated_messages(features)
        return self.update(nodes_representation, aggregated_messages)

# Define a custom Graph LSTM layer


class TrafficGraphLSTM(layers.Layer):
    def __init__(
        self,
        in_features,
        out_features,
        lstm_units: int,
        input_seq_len: int,
        output_seq_len: int,
        graph_info: TrafficGraphInfo,
        graph_conv_params: typing.Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if graph_conv_params is None:
            graph_conv_params = {
                "aggregation_type": "mean",
                "combination_type": "concat",
                "activation": None,
            }
        self.graph_conv = TrafficGraphConvolution(
            in_features, out_features, graph_info, **graph_conv_params)
        self.lstm = layers.LSTM(lstm_units, activation="relu")
        self.dense = layers.Dense(output_seq_len)
        self.input_seq_len, self.output_seq_len = input_seq_len, output_seq_len

    def call(self, inputs):
        inputs = tf.transpose(inputs, [
                              2, 0, 1, 3])  # Convert shape to (num_nodes, batch_size, input_seq_len, in_features)
        gcn_out = self.graph_conv(
            inputs
        )  # gcn_out has shape: (num_nodes, batch_size, input_seq_len, out_features)
        shape = tf.shape(gcn_out)
        num_nodes, batch_size, input_seq_len, out_features = (
            shape[0],
            shape[1],
            shape[2],
            shape[3],
        )
        gcn_out = tf.reshape(
            gcn_out, (batch_size * num_nodes, input_seq_len, out_features))
        lstm_out = self.lstm(
            gcn_out
        )  # lstm_out has shape: (batch_size * num_nodes, lstm_units)
        dense_output = self.dense(
            lstm_out
        )  # dense_output has shape: (batch_size * num_nodes, output_seq_len)
        output = tf.reshape(
            dense_output, (num_nodes, batch_size, self.output_seq_len))
        return tf.transpose(
            output, [1, 2, 0]
        )  # returns Tensor of shape (batch_size, output_seq_len, num_nodes)


# Model parameters
in_features = 1
batch_size = 64
epochs = 20
input_sequence_length = 12
forecast_horizon = 3
multi_horizon = False
out_features = 10
lstm_units = 64
graph_conv_params = {
    "aggregation_type": "mean",
    "combination_type": "concat",
    "activation": None,
}

# Create the Traffic LSTM-GCN model
traffic_lstm_gcn = TrafficGraphLSTM(
    in_features,
    out_features,
    lstm_units,
    input_sequence_length,
    forecast_horizon,
    graph_info,
    graph_conv_params,
)
inputs = layers.Input(
    (input_sequence_length, graph_info.num_nodes, in_features))
outputs = traffic_lstm_gcn(inputs)
model = keras.models.Model(inputs, outputs)

# Compile and train the model
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=0.0002),
    loss=keras.losses.MeanSquaredError(),
)
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=[keras.callbacks.EarlyStopping(patience=10)],
)

# Evaluate the model on the test dataset
x_test, y = next(test_dataset.as_numpy_iterator())
y_pred = model.predict(x_test)

# Visualize the results
plt.figure(figsize=(18, 6))
plt.plot(y[:, 0, 0])
plt.plot(y_pred[:, 0, 0])
plt.legend(["Actual", "Forecast"])

# Calculate Mean Squared Error
naive_mse, model_mse = (
    np.square(x_test[:, -1, :, 0] - y[:, 0, :]).mean(),
    np.square(y_pred[:, 0, :] - y[:, 0, :]).mean(),
)
print(f"Naive MSE: {naive_mse}, Model MSE: {model_mse}")

# Create an array to store predicted time series
predicted_time_series = []

# Iterate over each time series in x_test
for i in range(x_test.shape[2]):
    # Extract a single time series from x_test
    single_time_series = x_test[:, :, i:i+1]

    # Check if the sequence is long enough to make predictions
    if single_time_series.shape[1] >= input_sequence_length + forecast_horizon:
        # Predict using the model
        y_pred_single = model.predict(single_time_series)
        # Append the predicted time series to the array
        predicted_time_series.append(y_pred_single)

# Convert the list of predicted time series to a NumPy array
predicted_time_series = np.array(predicted_time_series)

# Number of time series and their lengths
num_time_series = x_test.shape[2]
time_series_length = x_test.shape[1]

# Create a plot to display all time series
plt.figure(figsize=(12, 6))
for i in range(len(predicted_time_series)):
    plt.plot(range(time_series_length),
             x_test[0, :, i], label=f'Test Series {i} (Actual)')
    plt.plot(range(time_series_length, time_series_length + forecast_horizon),
             predicted_time_series[i][0, :, 0], label=f'Test Series {i} (Predicted)')

plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend(loc='best')
plt.title('Actual vs. Predicted Time Series')
plt.grid(True)
plt.show()
