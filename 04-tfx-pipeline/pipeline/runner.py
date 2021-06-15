# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Local runner configuration"""


from absl import app
from absl import flags
from absl import logging

from aiplatform.pipelines import client as caippc

from tfx.dsl.components.base import executor_spec
from tfx.components.trainer import executor as trainer_executor
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor

from tfx.orchestration import data_types
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.orchestration.metadata import sqlite_metadata_connection_config

from tfx.proto import trainer_pb2

import pipeline 


def _compile_pipeline(pipeline_def, 
                     project_id,
                     pipeline_name,
                     pipeline_image,
                     pipeline_spec_path):
    """Compiles the pipeline."""

    # Create Kubeflow V2 runner
    runner_config = kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
        project_id=project_id,
        display_name=pipeline_name,
        default_image=pipeline_image)

    runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(
        config=runner_config,
        output_filename=pipeline_spec_path)

    # Compile the pipeline
    runner.run(pipeline_def)

def _submit_pipeline_run(
    project_id,
    region,
    api_key,
    pipeline_spec_path,
    pipeline_root,
    parameter_values=None):
    "Submits a run to AI Platform Pipelines."

    # Create AI Platform Pipelines client
    caipp_client = caippc.Client(
        project_id=project_id,
        region=region,
        api_key=api_key
    )

    # Submit a run
    caipp_client.create_run_from_job_spec(
        job_spec_path=pipeline_spec_path,
        pipeline_root=pipeline_root,
        parameter_values=parameter_values
    )


FLAGS = flags.FLAGS

# Runner settings
flags.DEFINE_string('pipeline_spec_path', 'pipeline.json', 'Pipeline spec path')
flags.DEFINE_bool('compile_only', False, 'Compile the pipeline but do not submit a run')
flags.DEFINE_bool('use_cloud_pipelines', False, 'Use AI Platform Pipelines')
flags.DEFINE_bool('use_cloud_executors', False, 'Use AI Platform and Dataflow for executors')
flags.DEFINE_string('api_key', 'None', 'API Key')
flags.mark_flag_as_required('api_key')
flags.DEFINE_string('sql_lite_path', '/home/jupyter/sqllite/metadata.sqlite', 'Path for SQL Lite')

# Pipeline compile time settings
flags.DEFINE_string('pipeline_name', 'covertype-training', 'Pipeline name')
flags.DEFINE_string('pipeline_image', 'gcr.io/jk-mlops-dev/covertype-tfx', 'Pipeline container image')
flags.DEFINE_string('model_name', 'convertype_classifier', 'Model display name')
flags.DEFINE_integer('train_steps', 1000, 'Training steps')
flags.DEFINE_integer('eval_steps', 500, 'Evaluation steps')
flags.DEFINE_string('project_id', 'jk-mlops-dev', 'Project ID')
flags.DEFINE_string('region', 'us-central1', 'Region')
flags.DEFINE_integer('dataflow_disk_size', 50, 'Dataflow worker disk size')
flags.DEFINE_string('dataflow_machine_type', 'e2-standard-8', 'Dataflow machine type')
flags.DEFINE_string('dataflow_temp_location', 'gs://jk-techsummit-bucket/dataflow-temp', 'Dataflow temp location')
flags.DEFINE_string('serving_model_uri', 'gs://jk-techsummit-bucket/models/covertype', 'Serving model dir')

# Runtime parameters
flags.DEFINE_string('data_root_uri', 'gs://workshop-datasets/covertype/small', 'Data root')
flags.DEFINE_string('schema_folder_uri', 'gs://jk-techsummit-bucket/schema', 'Schema folder uri')
flags.DEFINE_string('pipeline_root', None, 'Pipeline root')
flags.mark_flag_as_required('pipeline_root')


def main(argv):
    del argv
    
    # Overwrite the use_cloud_pipelines flag if the compile_only flag set
    if FLAGS.compile_only:
        FLAGS.use_cloud_pipelines = True

    # Config executors
    if FLAGS.use_cloud_executors:
        ai_platform_training_args = {
            'project': FLAGS.project_id,
            'region': FLAGS.region,
            'masterConfig': {
                'imageUri': FLAGS.pipeline_image,
            }
        }
        trainer_custom_config = {
             ai_platform_trainer_executor.TRAINING_ARGS_KEY: ai_platform_training_args}
        trainer_custom_executor_spec=executor_spec.ExecutorClassSpec(
            ai_platform_trainer_executor.GenericExecutor)

        beam_pipeline_args = [ 
            '--runner=DataflowRunner',
            '--experiments=shuffle_mode=auto',
            '--project=' + FLAGS.project_id,
            '--temp_location=' + FLAGS.dataflow_temp_location,
            '--disk_size_gb=' + str(FLAGS.dataflow_disk_size),
            '--machine_type=' + FLAGS.dataflow_machine_type,
            '--region=' + FLAGS.region ]
    else:
        trainer_custom_config = None
        trainer_custom_executor_spec=executor_spec.ExecutorClassSpec(
            trainer_executor.GenericExecutor)

        beam_pipeline_args = [
            '--direct_running_mode=multi_processing',
            # 0 means auto-detect based on on the number of CPUs available
            # during execution time.
            '--direct_num_workers=0' ] 

    # Config pipeline orchestrator
    if FLAGS.use_cloud_pipelines:
        metadata_connection_config = None
        data_root_uri = data_types.RuntimeParameter( 
            name='data-root-uri',
            ptype=str,
            default=FLAGS.data_root_uri)
        schema_folder_uri = data_types.RuntimeParameter(
            name='schema-folder-uri',
            ptype=str,
            default=FLAGS.schema_folder_uri)
    else:
        metadata_connection_config = (
           sqlite_metadata_connection_config(FLAGS.sql_lite_path) 
        )
        data_root_uri = FLAGS.data_root_uri
        schema_folder_uri = FLAGS.schema_folder_uri

    # Create the pipeline
    pipeline_def = pipeline.create_pipeline(
        pipeline_name=FLAGS.pipeline_name,
        pipeline_root=FLAGS.pipeline_root,
        serving_model_uri=FLAGS.serving_model_uri,
        data_root_uri=data_root_uri,
        schema_folder_uri=schema_folder_uri,
        eval_steps=FLAGS.eval_steps,
        train_steps=FLAGS.train_steps,
        trainer_custom_executor_spec=trainer_custom_executor_spec,
        trainer_custom_config=trainer_custom_config,
        beam_pipeline_args=beam_pipeline_args,
        metadata_connection_config=metadata_connection_config)

    # Run or compile the pipeline
    if FLAGS.use_cloud_pipelines:
        logging.info(f'Compiling pipeline to: {FLAGS.pipeline_spec_path}')
        _compile_pipeline(
            pipeline_def=pipeline_def,
            project_id=FLAGS.project_id,
            pipeline_name=FLAGS.pipeline_name,
            pipeline_image=FLAGS.pipeline_image,
            pipeline_spec_path=FLAGS.pipeline_spec_path
        )
        if FLAGS.compile_only:
            return

        # Set runtime parameters
        parameter_values = {
            'data-root-uri': FLAGS.data_root_uri,
            'schema-folder-uri': FLAGS.schema_folder_uri,
        }

        # Submit the run
        logging.info('Submitting AI Platform Pipelines job ...')
        _submit_pipeline_run(
            project_id=FLAGS.project_id,
            region=FLAGS.region,
            api_key=FLAGS.api_key,
            pipeline_spec_path=FLAGS.pipeline_spec_path,
            pipeline_root=FLAGS.pipeline_root,
            parameter_values=parameter_values)
    else:
        logging.info('Using local dag runner')
        LocalDagRunner().run(pipeline_def)

if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)

 



