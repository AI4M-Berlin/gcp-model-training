# gcp-model-training
Example repository to train a simple linear on Vertex AI

Prerequisites(see https://cloud.google.com/vertex-ai/docs/training/create-custom-job#configure_distributed_training):
1. Install Docker Engine
2. Configure Docker Engine to run it without sudo
3. Authenticate gcloud with `gcloud auth login`

To send a training job to Vertex AI run:

    `bash run_job.sh`

This will build a Docker image for training, upload it to GCP Container Registry and use the image to initialize training on Vertex AI which:
 - Downloads the training data (single .csv file) from Cloud Storage
 - Fits linear regression on this data
 - Uploads model to trained-models-temp bucket on Cloud Storage
