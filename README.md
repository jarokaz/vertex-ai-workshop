# Vertex AI Workshop

Hands-on labs introducing core GCP Vertex AI features.


## Environment Setup



### Vertex AI Notebook


#### Installing the required Python packages

In JupyterLab, open a terminal and install the following packages:

```
pip install --user google-cloud-aiplatform
pip install --user kfp
pip install --user google-cloud-pipeline-components==0.1.1
pip install --user google-cloud-bigquery-datatransfer
```

##### Creating a Tensorboard instance

Each participant will use their own Vertex Tensorboard instance. From the JupyterLab terminal:

```
export PREFIX=[YOUR PREFIX]
export PROJECT=phc-rdi-vertexai-sb-c50d9101
export REGION=us-central1
export DISPLAY_NAME=${PREFIX}-tensorboard

gcloud beta ai tensorboards create --display-name $DISPLAY_NAME \
  --project $PROJECT --region $REGION

```

Save the tensorboard name returned by the command as it will be needed when configuring the workshop notebooks.

You can get it at any time by listing Tensorboards in the project

```
gcloud beta ai tensorboards list \
  --project $PROJECT --region $REGION
```

##### Cloning the repo with hands-on labs
```
git clone https://github.com/jarokaz/vertex-ai-workshop
```

