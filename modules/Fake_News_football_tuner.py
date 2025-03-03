import os
import tensorflow as tf
import tensorflow_transform as tft
import keras_tuner as kt
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs
from keras_tuner.engine import base_tuner
from typing import NamedTuple, Dict, Text, Any
from keras_tuner import RandomSearch
from tfx.components import Tuner, Trainer
from tfx.proto import trainer_pb2

LABEL_KEY = 'real_news'
FEATURE_KEY = 'tweet'

# Utility function to rename transformed features
def transformed_name(key):
    return key + "_xf"

# Utility function to read gzip files
def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

# Input function for creating datasets
def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64):
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY)
    )
    return dataset

VOCAB_SIZE = 5000
SEQUENCE_LENGTH = 1500

vectorize_layer = layers.TextVectorization(
    standardize='lower_and_strip_punctuation',
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH
)

# Model builder for hyperparameter tuning
def model_builder(hp):
    """Build machine learning model"""
    embedding_dim = hp.Choice('embedding_dim', values=[32, 128])
    lstm_units = hp.Choice('lstm_units', values=[32, 128])
    dense_units = hp.Int('dense_units', min_value=32, max_value=128, step=16)
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.3, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])
    num_layers = hp.Int('num_layers', min_value=4, max_value=8, step=1)  # Added num_layers here

    inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)
    reshaped_narrative = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_narrative)
    x = layers.Embedding(VOCAB_SIZE, embedding_dim, name='embedding')(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units))(x)

    for _ in range(num_layers):  # Use num_layers here
        x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )
    return model

TunerFnResult = NamedTuple('TunerFnResult', [
    ('tuner', base_tuner.BaseTuner),
    ('fit_kwargs', Dict[Text, Any]),
])

early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor  = 'val_binary_accuracy',
    mode     = 'max',
    verbose  = 1,
    patience = 2
)

# Tuner function
def tuner_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = input_fn(fn_args.train_files[0], tf_transform_output, num_epochs=3, batch_size=64)
    val_set = input_fn(fn_args.eval_files[0], tf_transform_output, num_epochs=3, batch_size=64)

    vectorize_layer.adapt(
        [j[0].numpy()[0] for j in [
            i[0][transformed_name(FEATURE_KEY)] for i in list(train_set)
        ]]
    )

    model_tuner = RandomSearch(
        hypermodel=model_builder,
        objective=kt.Objective('val_binary_accuracy', direction='max'),
        max_trials=4,
        executions_per_trial=2,
        directory=fn_args.working_dir,
        project_name='suicide_text_random_tuner',
    )

    return TunerFnResult(
        tuner=model_tuner,
        fit_kwargs={
            'x': train_set,
            'validation_data': val_set,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps,
            'callbacks': [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_binary_accuracy',
                    mode='max',
                    patience=2,
                    verbose=1
                )
            ]
        }
    )
