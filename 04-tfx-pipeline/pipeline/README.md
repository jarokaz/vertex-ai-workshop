# Continuous training with TFX and Cloud AI Platform

This hands-on lab guides you through the process of implementing a TensorFlow Extended (TFX) continuous training pipeline that automates training and deployment of a TensorFlow 2.4 model using Cloud AI Platform (Unfied) Pipelines.

The below diagram represents the workflow orchestrated by the pipeline.

![TFX_CAIP](/images/tfx-ucaip.png).

1. Training data in the CSV format is ingested from a GCS location using *CsvExampleGen*. The URI to the data root is passed as a runtime parameter. The *CsvExampleGen* component splits the source data into training and evaluation splits and converts the data into the TFRecords format.
2. The *StatisticsGen* component generates statistics for both splits.
3. The *SchemaGen* component autogenerates a schema . This is done tracking and lineage analysis. The pipeline uses a curated schema imported by the *ImportedNode* component.
4. The *ImporterNode* component is used to bring the curated schema file into the pipeline. The location of the schema file is passed as a runtime parameter. 
5. The *ExampleValidator* component validates the generated examples against the imported schema
6. The *Transform* component preprocess the data to the format required by the *Trainer* component. It also saves the preprocessing TensorFlow graph for consistent feature engineering at training and serving time.
7. The *Trainer* starts an AI Platform Training job. The AI Platform Training job is configured for training in a custom container. 
8. The *Tuner* component in the example pipeline tunes model hyperparameters using CloudTuner (KerasTuner instance) and AI Platform Vizier as a back-end. It can be added and removed from the pipeline using the `enable_tuning` environment variable set in the notebook or in the pipeline code. When included in the pipeline, it ouputs a "best_hyperparameter" artifact directly into the *Trainer*. When excluded hyperparameters are drawn from the defaults set in the pipeline code.
9. The *ResolverNode* component retrieves the best performing model from the previous runs and passed it to the *Evaluator* to be used as a baseline during model validation.
10. The *Evaluator* component evaluates the trained model against the eval split and validates against the baseline model from the *ResolverNode*. If the new model exceeds validation thresholds it is marked as "blessed".
11. The *InfraValidator* component validates the model serving infrastructure and provides a "infra_blessing" that the model can be loaded and queried for predictions.
12. If the new model is blessed by the *Evaluator* and *InfraValidator*, the *Pusher* deploys the model to AI Platform Prediction.

The ML model utilized in the labs  is a multi-class classifier that predicts the type of  forest cover from cartographic data. The model is trained on the [Covertype Data Set](/datasets/covertype/README.md) dataset.


### Building a pipeline container

```
export PROJECT_ID=jk-mlops-dev
export IMAGE_NAME=gcr.io/$PROJECT_ID/covertype-tfx
gcloud builds submit --tag $IMAGE_NAME .
```

### Creating a regional bucket

```
BUCKET_NAME=gs://jk-techsummit-bucket
REGION=us-central1

gsutil mb -l $REGION $BUCKET_NAME
```