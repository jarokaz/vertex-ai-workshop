

# Copyright 2021 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import json
import os
import tensorflow as tf
import tensorflow_data_validation as tfdv

from absl import app
from absl import flags
from absl import logging

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow_io import bigquery as tfio_bq


FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 3, 'Nubmer of epochs')
flags.DEFINE_integer('units', 32, 'Number units in a hidden layer')
flags.DEFINE_integer('per_replica_batch_size', 128, 'Per replica batch size')
flags.DEFINE_float('dropout_ratio', 0.5, 'Dropout ratio')
flags.DEFINE_string('training_table', None, 'Training table name')
flags.DEFINE_string('validation_table', None, 'Validationa table name')
flags.DEFINE_string('schema_file', None, 'Location of the data schema file')
flags.mark_flag_as_required('training_table')
flags.mark_flag_as_required('validation_table')
flags.mark_flag_as_required('schema_file')

LOCAL_MODEL_DIR = '/tmp/saved_model'
LOCAL_TB_DIR = '/tmp/logs'
LOCAL_CHECKPOINT_DIR = '/tmp/checkpoints'
TARGET_TAG = 'target'


def schema_to_features(schema):
    """Converts schema_pb2 protobuf to feature dictionary."""
    
    features = {}
    for feature in schema.feature:
        if feature.type == 2:
            if feature.int_domain.is_categorical:
                features[feature.name] = ('categorical', tf.int64)
            else:
                features[feature.name] = ('numeric', tf.int64)
        elif feature.type == 1:
            features[feature.name] = ('categorical', tf.string)
        elif feature.type == 3:
            features[feature.name] = ('numeric', tf.double)
    
    return features

def get_target_feature(schema):
    """Returns the name of the target feature from schema."""
    
    target_feature = None
    for feature in schema.feature:
        if feature.HasField('annotation'):
            if TARGET_TAG in feature.annotation.tag:
                target_feature = feature.name
    return target_feature


def set_job_dirs():
    """Sets job directories based on env variables set by Vertex AI."""
    
    model_dir = os.getenv('AIP_MODEL_DIR', LOCAL_MODEL_DIR)
    tb_dir = os.getenv('AIP_TENSORBOARD_LOG_DIR', LOCAL_TB_DIR)
    checkpoint_dir = os.getenv('AIP_CHECKPOINT_DIR', LOCAL_CHECKPOINT_DIR)
    
    return model_dir, tb_dir, checkpoint_dir



def get_bq_dataset(table_name, features, target_feature, batch_size=32):
    
    def _transform_row(row_dict):
        trimmed_dict = {column:
                       (tf.strings.strip(tensor) if tensor.dtype == 'string' else tensor) 
                       for (column,tensor) in row_dict.items()
                       }
        target = trimmed_dict.pop(target_feature)
        return (trimmed_dict, target)
    
    selected_fields = {key: {'output_type': value[1]} 
                         for key, value in features.items()}
    project_id, dataset_id, table_id = table_name.split('.')
    client = tfio_bq.BigQueryClient()
    parent = f'projects/{project_id}'

    read_session = client.read_session(
        parent=parent,
        project_id=project_id,
        table_id=table_id,
        dataset_id=dataset_id,
        selected_fields=selected_fields,
    )

    dataset = read_session.parallel_read_rows().map(_transform_row).batch(batch_size)
    
    return dataset


def get_category_encoding_layer(name, dataset, dtype):
    """Creates a CategoryEncoding layer for a given feature."""

    if dtype == tf.string:
      index = preprocessing.StringLookup()
    else:
      index = preprocessing.IntegerLookup()

    feature_ds = dataset.map(lambda x, y: x[name])
    index.adapt(feature_ds)
    encoder = preprocessing.CategoryEncoding(max_tokens=index.vocab_size())

    return lambda feature: encoder(index(feature))


def get_normalization_layer(name, dataset):
    """"Creates a Normalization layer for a given feature."""
    normalizer = preprocessing.Normalization()

    feature_ds = dataset.map(lambda x, y: x[name])
    normalizer.adapt(feature_ds)

    return normalizer


def create_model(dataset, input_features, units, dropout_ratio):
    """Creates a binary classifier for Chicago Taxi tip prediction task."""
    
    all_inputs = []
    encoded_features = []
    for feature_name, feature_info in input_features.items():
        col = tf.keras.Input(shape=(1,), name=feature_name, dtype=feature_info[1])
        if feature_info[0] == 'categorical':
            
            encoding_layer = get_category_encoding_layer(feature_name, 
                                                         dataset,
                                                         feature_info[1])
        else:
            encoding_layer = get_normalization_layer(feature_name,
                                                     dataset) 
        encoded_col = encoding_layer(col)
        all_inputs.append(col)
        encoded_features.append(encoded_col)
        
    all_features = tf.keras.layers.concatenate(encoded_features)
    
    x = tf.keras.layers.Dense(units, activation="relu")(all_features)
    x = tf.keras.layers.Dropout(dropout_ratio)(x)
    output = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(all_inputs, output)
    
    return model


def main(argv):
    del argv
    
    # Set distribution strategy
    strategy = tf.distribute.MirroredStrategy()
    
    global_batch_size = (strategy.num_replicas_in_sync *
                         FLAGS.per_replica_batch_size)
    
    schema = tfdv.load_schema_text(FLAGS.schema_file)
    
    features = schema_to_features(schema)
    target_feature = get_target_feature(schema)

    if not target_feature:
        raise RuntimeError('Schema does not have target feature annotation')
    
    # Prepare datasets
    validation_ds = get_bq_dataset(FLAGS.validation_table, 
                                   features,
                                   target_feature,
                                   batch_size=global_batch_size)
    training_ds = get_bq_dataset(FLAGS.training_table,
                                 features,
                                 target_feature,
                                 batch_size=global_batch_size)
    
    # Prepare the model
    logging.info('Creating the model ...')
    input_features = {key: value for key, value in features.items() if key != target_feature}
    
    with strategy.scope():
        model = create_model(training_ds, input_features, FLAGS.units, FLAGS.dropout_ratio)
        model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    # Configure Keras callbacks
    model_dir, tb_dir, checkpoint_dir = set_job_dirs()
    callbacks = [tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=checkpoint_dir)]
    callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=tb_dir, update_freq='batch'))
    
    logging.info('Starting training ...')
    model.fit(training_ds, 
              epochs=FLAGS.epochs, 
              validation_data=validation_ds,
              callbacks=callbacks)
    
    # Save trained model
    logging.info('Training completed. Saving the trained model to: {}'.format(model_dir))
    model.save(model_dir)  
    
    
if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
