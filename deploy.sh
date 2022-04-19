#!/bin/bash

# Path to the model (will also be the path on AWS S3)
MODEL_NAME="piano_event_performer_2021-10-01_16:03:06"

# Name of the CIA release on Github
RELEASE_NAME="1.0-alpha"

# --- Push to S3

# echo "Pushing ${MODEL_NAME} to S3"
# aws s3api put-object --bucket ghadjeres --key "${MODEL_NAME}/overfitted/model" --body "models/${MODEL_NAME}/overfitted/model" --no-cli-pager
# aws s3api put-object --bucket ghadjeres --key "${MODEL_NAME}/early_stopped/model" --body "models/${MODEL_NAME}/early_stopped/model" --no-cli-pager
# aws s3api put-object --bucket ghadjeres --key "${MODEL_NAME}/config.py" --body "models/${MODEL_NAME}/config.py" --no-cli-pager

# echo "Granting public permission"
# aws s3api put-object-acl --bucket ghadjeres --key "${MODEL_NAME}/overfitted/model" --acl public-read --no-cli-pager
# aws s3api put-object-acl --bucket ghadjeres --key "${MODEL_NAME}/early_stopped/model" --acl public-read --no-cli-pager
# aws s3api put-object-acl --bucket ghadjeres --key "${MODEL_NAME}/config.py" --acl public-read --no-cli-pager

# echo 'Building Docker'
# docker build -t piano_inpainting_app:v3 --build-arg GITHUB_TOKEN="$(cat /home/gaetan/.secrets/github_token)" --build-arg SSH_PRIVATE_KEY="$(cat /home/gaetan/.ssh/id_rsa)" --build-arg AWS_BUCKET_NAME="${MODEL_NAME}" --build-arg RELEASE_NAME="${RELEASE_NAME}" .  
# add --no-cache flag if necessary

# --- Push to ECR
echo 'Pushing to ECR'
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 854428102214.dkr.ecr.eu-west-1.amazonaws.com

docker tag piano_inpainting_app:v3 854428102214.dkr.ecr.eu-west-1.amazonaws.com/piano_inpainting_app:v3

docker push 854428102214.dkr.ecr.eu-west-1.amazonaws.com/piano_inpainting_app:v3
