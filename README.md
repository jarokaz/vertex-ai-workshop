# Vertex AI Workshop


## Environment Setup

### Install pre-requisites

```
pip install --user google-cloud-aiplatform
pip install --user kfp
pip install --user google-cloud-pipeline-components
```

### Create a Tensorboard instance

```
PROJECT=jk-vertex-workshop
REGION=us-central1
PREFIX=jkvw
DISPLAY_NAME=${PREFIX}-tensorboard

gcloud beta ai tensorboards create --display-name $DISPLAY_NAME \
  --project $PROJECT --region $REGION

```

Save the tensorboard name returned by the command

#### List Tensorboards

```
gcloud beta ai tensorboards list \
  --project $PROJECT --region $REGION
```

## Lab notes

