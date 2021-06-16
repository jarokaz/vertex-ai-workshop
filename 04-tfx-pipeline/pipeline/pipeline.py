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
# limitations under the License.
"""Covertype training pipeline DSL."""


import tensorflow_model_analysis as tfma
import tfx

from ml_metadata.proto import metadata_store_pb2

from tfx.components import (
    Evaluator,
    CsvExampleGen,
    ExampleValidator,
    ImporterNode,
    Pusher,
    StatisticsGen,
    Trainer,
    Transform,
    SchemaGen,
)
    
from tfx.dsl.components.common.importer import Importer
from tfx.orchestration import pipeline
from tfx.orchestration import data_types
from tfx.orchestration.metadata import sqlite_metadata_connection_config

from tfx.proto import example_gen_pb2
from tfx.proto import pusher_pb2


from typing import Optional, Dict, List, Text, Union, Any

TRANSFORM_MODULE_FILE='preprocess.py'
TRAIN_MODULE_FILE='train.py'

def create_pipeline(
    pipeline_name: Text, 
    pipeline_root: Text,
    serving_model_uri: Text, 
    data_root_uri: Text,
    schema_folder_uri: Text, 
    train_steps: int,
    eval_steps: int,
    enable_cache: Optional[bool] = False,
) -> pipeline.Pipeline:

    """Trains and deploys the Keras Covertype Classifier with TFX and AI Platform Pipelines."""
  
    # Brings data into the pipeline and splits the data into training and eval splits
    output_config = example_gen_pb2.Output(
      split_config=example_gen_pb2.SplitConfig(splits=[
          example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=4),
          example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
      ]))
  
    examplegen = CsvExampleGen(
        input_base=data_root_uri,
        output_config=output_config
    ).with_id("TrainDataGen")
  
    # Computes statistics over data for visualization and example validation.
    statisticsgen = StatisticsGen(
        examples=examplegen.outputs.examples
    ).with_id("StatisticsGen")
  
    # Generates schema based on statistics files. Even though, we use user-provided schema
    # we still want to generate the schema of the newest data for tracking and comparison
    schemagen = SchemaGen(
        statistics=statisticsgen.outputs.statistics
    ).with_id("SchemaGen")
  
    # Import a user-provided schema
    import_schema = ImporterNode(
        source_uri=schema_folder_uri,
        artifact_type=tfx.types.standard_artifacts.Schema
    ).with_id("SchemaImporter")
  
    # Performs anomaly detection based on statistics and data schema.
    examplevalidator = ExampleValidator(
        statistics=statisticsgen.outputs.statistics, 
        schema=import_schema.outputs.result
    ).with_id("ExampleValidator")
  
    # Performs transformations and feature engineering in training and serving.
    transform = Transform(
        examples=examplegen.outputs.examples,
        schema=import_schema.outputs.result,
        module_file=TRANSFORM_MODULE_FILE
    ).with_id("DataTransformer")
  
    # Trains the model using a user provided trainer function.
    trainer = Trainer(
        module_file=TRAIN_MODULE_FILE,
        transformed_examples=transform.outputs.transformed_examples,
        schema=import_schema.outputs.result,
        transform_graph=transform.outputs.transform_graph,     
        train_args={'num_steps': train_steps},
        eval_args={'num_steps': eval_steps},
    ).with_id("Trainer")
  
  
    # Uses TFMA to compute a evaluation statistics over features of a model.
    accuracy_threshold = tfma.MetricThreshold(
                  value_threshold=tfma.GenericValueThreshold(
                      lower_bound={'value': 0.5},
                      upper_bound={'value': 0.99}),
                  )
  
    metrics_specs = tfma.MetricsSpec(
                     metrics = [
                         tfma.MetricConfig(class_name='SparseCategoricalAccuracy',
                             threshold=accuracy_threshold),
                         tfma.MetricConfig(class_name='ExampleCount')])
  
    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(label_key='Cover_Type')
        ],
        metrics_specs=[metrics_specs],
        slicing_specs=[
            tfma.SlicingSpec(),
            tfma.SlicingSpec(feature_keys=['Wilderness_Area'])
        ]
    )
    
    evaluator = Evaluator(
        examples=examplegen.outputs.examples,
        model=trainer.outputs.model,
        eval_config=eval_config
    ).with_id("ModelEvaluator")
  
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_uri)))
  
    components=[
        examplegen, 
        statisticsgen,
        schemagen,      
        import_schema,
        examplevalidator,
        transform,
        trainer, 
        evaluator, 
        pusher 
    ]
  
    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=enable_cache
    )
  
  
  