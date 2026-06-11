"""Deep-learning model definitions."""

from __future__ import annotations

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, LSTM, MaxPooling1D
from tensorflow.keras.optimizers import Adam


def build_lstm(sequence_length: int, n_features: int) -> Sequential:
    model = Sequential([
        LSTM(64, input_shape=(sequence_length, n_features), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def build_cnn(sequence_length: int, n_features: int) -> Sequential:
    model = Sequential([
        Conv1D(64, kernel_size=3, activation="relu", input_shape=(sequence_length, n_features)),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=3, activation="relu"),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def build_cnn_lstm(sequence_length: int, n_features: int) -> Sequential:
    model = Sequential([
        Conv1D(64, kernel_size=3, activation="relu", input_shape=(sequence_length, n_features)),
        MaxPooling1D(pool_size=2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model
