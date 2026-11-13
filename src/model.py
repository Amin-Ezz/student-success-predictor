"""
Multi-Task Deep Learning Model Definition.
Constructs a unified deep neural network with shared representation layers
and three task-specific output heads:
  1. future_gpa: Continuous regression (linear)
  2. dropout: Binary classification (sigmoid)
  3. performance: Multi-class 4-level classification (softmax)
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization


def build_multi_task_model(
    input_dim: int,
    learning_rate: float = 0.001,
) -> Model:
    """
    Build and compile a multi-task learning neural network.
    
    Parameters
    ----------
    input_dim : int
        Number of input features.
    learning_rate : float, default=0.001
        Adam optimizer learning rate.
        
    Returns
    -------
    Model
        Compiled tf.keras.Model with 3 task-specific heads.
    """
    # Shared Representation Backbone
    input_layer = Input(shape=(input_dim,), name="student_features_input")

    dense_1 = Dense(256, activation="relu", name="shared_dense_1")(input_layer)
    batch_norm_1 = BatchNormalization(name="shared_bn_1")(dense_1)
    dropout_1 = Dropout(0.30, name="shared_dropout_1")(batch_norm_1)

    dense_2 = Dense(128, activation="relu", name="shared_dense_2")(dropout_1)
    batch_norm_2 = BatchNormalization(name="shared_bn_2")(dense_2)
    dropout_2 = Dropout(0.25, name="shared_dropout_2")(batch_norm_2)

    dense_3 = Dense(64, activation="relu", name="shared_dense_3")(dropout_2)
    batch_norm_3 = BatchNormalization(name="shared_bn_3")(dense_3)
    dropout_3 = Dropout(0.20, name="shared_dropout_3")(batch_norm_3)

    shared_features = Dense(32, activation="relu", name="shared_dense_4")(dropout_3)

    # Task 1: Future GPA Head (Regression)
    gpa_branch = Dense(32, activation="relu", name="gpa_dense_1")(shared_features)
    gpa_branch = Dense(16, activation="relu", name="gpa_dense_2")(gpa_branch)
    future_gpa_output = Dense(1, activation="linear", name="future_gpa")(gpa_branch)

    # Task 2: Dropout Risk Head (Binary Classification)
    dropout_branch = Dense(32, activation="relu", name="dropout_dense_1")(shared_features)
    dropout_branch = Dense(16, activation="relu", name="dropout_dense_2")(dropout_branch)
    dropout_output = Dense(1, activation="sigmoid", name="dropout")(dropout_branch)

    # Task 3: Performance Tier Head (Multi-Class Classification)
    perf_branch = Dense(32, activation="relu", name="perf_dense_1")(shared_features)
    perf_branch = Dense(16, activation="relu", name="perf_dense_2")(perf_branch)
    performance_output = Dense(4, activation="softmax", name="performance")(perf_branch)

    # Multi-Output Model
    model = Model(
        inputs=input_layer,
        outputs=[future_gpa_output, dropout_output, performance_output],
        name="StudentMultiTaskNet",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "future_gpa": "mse",
            "dropout": "binary_crossentropy",
            "performance": "sparse_categorical_crossentropy",
        },
        loss_weights={
            "future_gpa": 1.0,
            "dropout": 1.0,
            "performance": 1.5,
        },
        metrics={
            "future_gpa": ["mae"],
            "dropout": ["accuracy"],
            "performance": ["accuracy"],
        },
    )

    return model
