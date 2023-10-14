import pandas as pd
import numpy as np
import os
import typing
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import osmnx as ox
import networkx as nx
import geopandas as gpd

# Step 1: Load Map Data


def load_map_data():
    # Load map data for a specific region
    map_graph = ox.graph_from_bbox(
        35.7209, 35.7042, 51.4215, 51.3856, network_type='drive')
    return map_graph

# Step 2: Visualize the Map


def visualize_map(map_graph):
    # Plot the road network
    fig, ax = ox.plot_graph(map_graph)
    plt.show()

# Step 3: Load Traffic Data


def load_traffic_data():
    # Download and extract the traffic dataset
    traffic_data_url = "https://github.com/VeritasYin/STGCN_IJCAI-18/raw/master/dataset/PeMSD7_Full.zip"
    data_dir = keras.utils.get_file(
        origin=traffic_data_url, extract=True, archive_format="zip")
    data_dir = data_dir.rstrip("PeMSD7_Full.zip")

    # Load route distances and speeds data
    route_distances = pd.read_csv(os.path.join(
        data_dir, "PeMSD7_W_228.csv"), header=None).to_numpy()
    speeds_array = pd.read_csv(os.path.join(
        data_dir, "PeMSD7_V_228.csv"), header=None).to_numpy()

    return route_distances, speeds_array

# Step 4: Visualize Traffic Data


def visualize_traffic_data(speeds_array):
    # Visualize traffic data for selected routes
    sample_routes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for route in sample_routes:
        plt.plot(speeds_array[:, route])

    plt.legend([f"Route {route}" for route in sample_routes])
    plt.xlabel("Time Steps")
    plt.ylabel("Speed")
    plt.show()

# Step 5: Data Preprocessing


def preprocess_data(data_array, train_size, val_size):
    # Split data into train, validation, and test sets and normalize it
    num_time_steps = data_array.shape[0]
    num_train, num_val = int(
        num_time_steps * train_size), int(num_time_steps * val_size)
    train_array = data_array[:num_train]
    mean, std = train_array.mean(axis=0), train_array.std(axis=0)
    train_array = (train_array - mean) / std
    val_array = (data_array[num_train:(num_train + num_val)] - mean) / std
    test_array = (data_array[(num_train + num_val):] - mean) / std
    return train_array, val_array, test_array

# Step 6: Create TensorFlow Datasets


def create_tf_datasets(data_array, input_sequence_length, forecast_horizon, batch_size, shuffle=True, multi_horizon=True):
    # Create TensorFlow datasets for input and target sequences
    inputs = timeseries_dataset_from_array(
        np.expand_dims(data_array[:-forecast_horizon], axis=-1),
        None,
        sequence_length=input_sequence_length,
        shuffle=False,
        batch_size=batch_size,
    )
    target_offset = input_sequence_length if multi_horizon else input_sequence_length + \
        forecast_horizon - 1
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

# Step 7: Compute Adjacency Matrix


def compute_adjacency_matrix(route_distances, sigma2, epsilon):
    # Compute the adjacency matrix from the distances matrix
    num_routes = route_distances.shape[0]
    route_distances = route_distances / 10000.0
    w2, w_mask = (
        route_distances * route_distances,
        np.ones([num_routes, num_routes]) - np.identity(num_routes),
    )
    return (np.exp(-w2 / sigma2) >= epsilon) * w_mask

# Step 8: Define Graph Information


class GraphInfo:
    def __init__(self, edges, num_nodes):
        self.edges = edges
        self.num_nodes = num_nodes

# Step 9: Define Graph Convolution Layer


class GraphConvolution(layers.Layer):
    def __init__(self, in_feat, out_feat, graph_info, aggregation_type="mean", combination_type="concat", activation=None, **kwargs):
        super().__init__(**kwargs)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.graph_info = graph_info
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.weight = tf.Variable(
            initial_value=keras.initializers.glorot_uniform()(
                shape=(in_feat, out_feat), dtype="float32"),
            trainable=True,
        )
        self.activation = layers.Activation(activation)

    def aggregate(self, neighbour_representations):
        aggregation_func = {
            "sum": tf.math.unsorted_segment_sum,
            "mean": tf.math.unsorted_segment_mean,
            "max": tf.math.unsorted_segment_max,
        }.get(self.aggregation_type)
        if aggregation_func:
            return aggregation_func(
                neighbour_representations,
                self.graph_info.edges[0],
                num_segments=self.graph_info.num_nodes,
            )
        raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")

    def compute_nodes_representation(self, features):
        return tf.matmul(features, self.weight)

    def compute_aggregated_messages(self, features):
        neighbour_representations = tf.gather(
            features, self.graph_info.edges[1])
        aggregated_messages = self.aggregate(neighbour_representations)
        return tf.matmul(aggregated_messages, self.weight)

    def update(self, nodes_representation, aggregated_messages):
        if self.combination_type == "concat":
            h = tf.concat([nodes_representation, aggregated_messages], axis=-1)
        elif self.combination_type == "add":
            h = nodes_representation + aggregated_messages
        else:
            raise ValueError(
                f"Invalid combination type: {self.combination_type}.")
        return self.activation(h)

    def call(self, features):
        nodes_representation = self.compute_nodes_representation(features)
        aggregated_messages = self.compute_aggregated_messages(features)
        return self.update(nodes_representation, aggregated_messages)

# Step 10: Define Graph LSTM Layer


class GraphLSTM(layers.Layer):
    def __init__(self, in_feat, out_feat, lstm_units, input_seq_len, output_seq_len, graph_info, graph_conv_params=None, **kwargs):
        super().__init__(**kwargs)
        if graph_conv_params is None:
            graph_conv_params = {
                "aggregation_type": "mean",
                "combination_type": "concat",
                "activation": None,
            }
        self.graph_conv = GraphConvolution(
            in_feat, out_feat, graph_info, **graph_conv_params)
        self.lstm = layers.LSTM(lstm_units, activation="relu")
        self.dense = layers.Dense(output_seq_len)
        self.input_seq_len, self.output_seq_len = input_seq_len, output_seq_len

    def call(self, inputs):
        inputs = tf.transpose(inputs, [2, 0, 1, 3])
        gcn_out = self.graph_conv(inputs)
        shape = tf.shape(gcn_out)
        num_nodes, batch_size, input_seq_len, out_feat = (
            shape[0],
            shape[1],
            shape[2],
            shape[3],
        )
        gcn_out = tf.reshape(
            gcn_out, (batch_size * num_nodes, input_seq_len, out_feat))
        lstm_out = self.lstm(gcn_out)
        dense_output = self.dense(lstm_out)
        output = tf.reshape(
            dense_output, (num_nodes, batch_size, self.output_seq_len))
        return tf.transpose(output, [1, 2, 0])

# Step 11: Create and Train the LSTM-GCN Model


def create_and_train_lstm_gcn_model(input_shape, train_dataset, val_dataset, epochs):
    model = keras.models.Model(input_shape, output)

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.0002),
        loss=keras.losses.MeanSquaredError(),
    )
    model.fit(
        train_dataset,
        # Replace val_target with your actual validation target data
        validation_data=(val_dataset, val_target),
        epochs=epochs,
        callbacks=[keras.callbacks.EarlyStopping(patience=10)]
    )

    return model

# Step 12: Evaluate the Model


def evaluate_model(model, test_dataset):
    x_test, y = next(test_dataset.as_numpy_iterator())
    y_pred = model.predict(x_test)
    return x_test, y, y_pred

# Step 13: Visualize Time Series in x_test


def visualize_time_series(x_test):
    num_time_series = x_test.shape[0]
    time_steps = x_test.shape[1]
    num_roads = x_test.shape[2]

    fig, axes = plt.subplots(num_time_series, figsize=(12, 8*num_time_series))
    fig.suptitle("Time Series in x_test", fontsize=16)

    for i in range(num_time_series):
        ax = axes[i]
        ax.set_title(f"Time Series {i + 1}")
        for j in range(num_roads):
            ax.plot(x_test[i, :, j, 0], label=f"Road {j}")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Speed")
        ax.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()


if __name__ == "__main__":
    # Step 1: Load Map Data
    map_graph = load_map_data()

    # Step 2: Visualize the Map
    visualize_map(map_graph)

    # Step 3: Load Traffic Data
    route_distances, speeds_array = load_traffic_data()

    # Step 4: Visualize Traffic Data
    visualize_traffic_data(speeds_array)

    # Step 5: Data Preprocessing
    train_size, val_size = 0.5, 0.2
    train_array, val_array, test_array = preprocess_data(
        speeds_array, train_size, val_size)

    # Step 6: Create TensorFlow Datasets
    batch_size = 64
    input_sequence_length = 12
    forecast_horizon = 3
    multi_horizon = False
    train_dataset, val_dataset = create_tf_datasets(
        train_array, input_sequence_length, forecast_horizon, batch_size), val_array
    test_dataset = create_tf_datasets(test_array, input_sequence_length, forecast_horizon,
                                      test_array.shape[0], shuffle=False, multi_horizon=multi_horizon)

    # Step 7: Compute Adjacency Matrix
    sigma2 = 0.1
    epsilon = 0.5
    adjacency_matrix = compute_adjacency_matrix(
        route_distances, sigma2, epsilon)

    # Step 8: Define Graph Information
    node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
    graph = GraphInfo(edges=(node_indices.tolist(
    ), neighbor_indices.tolist()), num_nodes=adjacency_matrix.shape[0])

    # Step 9: Define Graph Convolution Layer
    graph_conv_params = {
        "aggregation_type": "mean",
        "combination_type": "concat",
        "activation": None,
    }

    # Step 10: Define Graph LSTM Layer
    in_feat = 1
    out_feat = 10
    lstm_units = 64
    lstm_gcn = GraphLSTM(in_feat, out_feat, lstm_units,
                         input_sequence_length, forecast_horizon, graph, graph_conv_params)

    # Step 11: Create and Train the LSTM-GCN Model
    input_shape = layers.Input(
        (input_sequence_length, graph.num_nodes, in_feat))
    output = lstm_gcn(input_shape)
    epochs = 20
    model = create_and_train_lstm_gcn_model(
        input_shape, train_dataset, val_dataset, epochs)

    # Step 12: Evaluate the Model
    x_test, y, y_pred = evaluate_model(model, test_dataset)

    # Step 13: Visualize Time Series in x_test
    visualize_time_series(x_test)
