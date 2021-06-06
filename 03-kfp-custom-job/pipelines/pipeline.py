# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""KFP pipeline definition."""

import os
import sys
import logging
import uuid

import kfp
from google.cloud import aiplatform
from google_cloud_pipeline_components import aiplatform as gcc_aip
from kfp.v2 import compiler
from kfp.v2.dsl import component
from kfp.v2.google import experimental
from kfp.v2.google.client import AIPlatformClient

import components

VERTEX_TRAINING_JOB_NAME = 'taxi-tip-predictor-training-job'
PIPELINE_NAME = 'taxi-tip-predictor-continuous-training'

@kfp.dsl.pipeline(name=PIPELINE_NAME)
def taxi_tip_predictor_pipeline(
    project: str,
    model_display_name: str,
    training_container_image_uri: str,
    epochs: int,
    per_replica_batch_size: int,
    training_table: str,
    validation_table: str,
    replica_count: str = '1',
    machine_type: str = 'n1-standard-4',
    accelerator_type: str = 'NVIDIA_TESLA_T4',
    accelerator_count: str = '1'  

):

#    train_task = components.training_op("model training")
#    experimental.run_as_aiplatform_custom_job(
#        train_task,
#        worker_pool_specs=[
#            {
#                "containerSpec": {
#                    "args": TRAINER_ARGS,
#                    "env": [{"name": "AIP_MODEL_DIR", "value": WORKING_DIR}],
#                    "imageUri": "gcr.io/google-samples/bw-cc-train:latest",
#                },
#                "replicaCount": "1",
#                "machineSpec": {
#                    "machineType": "n1-standard-16",
#                    "accelerator_type": aiplatform.gapic.AcceleratorType.NVIDIA_TESLA_K80,
#                    "accelerator_count": 2,
#                },
#            }
#        ],
#    )

    command = ["python", "train.py"]
    args = [
        '--epochs=' + str(epochs),
        '--per_replica_batch_size=' + str(per_replica_batch_size),
        '--validation_table=' + str(validation_table),
        '--training_table=' + str(training_table),
    ]

    train_model = gcc_aip.CustomContainerTrainingJobRunOp(
        project=project,
        display_name=VERTEX_TRAINING_JOB_NAME,
        container_uri=training_container_image_uri,
        args=args,
        command=command,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_type
    )
#    model_upload_op = gcc_aip.ModelUploadOp(
#        project=project,
#        display_name=model_display_name,
#        artifact_uri=WORKING_DIR,
#        serving_container_image_uri=serving_container_image_uri,
#        serving_container_environment_variables={"NOT_USED": "NO_VALUE"},
#    )
#    model_upload_op.after(train_task)
#
#    endpoint_create_op = gcc_aip.EndpointCreateOp(
#        project=project,
#        display_name="pipelines-created-endpoint",
#    )
#
#    model_deploy_op = gcc_aip.ModelDeployOp(  # noqa: F841
#        project=project,
#        endpoint=endpoint_create_op.outputs["endpoint"],
#        model=model_upload_op.outputs["model"],
#        deployed_model_display_name=model_display_name,
#        machine_type="n1-standard-4",
#    )
#
