DATA_PATH=gs://ai4m-tabd-1/heart.csv
MODEL_BUCKET=gs://trained-models-temp
REG=l2

REGION=europe-west3
PROJECT=ai4medicine-cloud
MODEL_NAME=first_model
MACHINE=n1-standard-4
SKLEARN_IMAGE=europe-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.0-23:latest
PACKAGE_PATH=/Users/antonshemyakov/code/gcp-model-training
MODULE=trainer.task

ARGS="--input_path=$DATA_PATH,\
--output_path=$MODEL_BUCKET,\
--regularization=l2"

gcloud ai custom-jobs create \
  --region=${REGION} \
  --project=${PROJECT} \
  --display-name=${MODEL_NAME} \
  --worker-pool-spec=machine-type=${MACHINE},replica-count=1,executor-image-uri=${SKLEARN_IMAGE},local-package-path=${PACKAGE_PATH},python-module=${MODULE} \
  --args="$ARGS"
