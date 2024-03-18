%%sh
docker build . --platform linux/amd64 -t sagemaker-build

export ECR_URL="<aws-account-id>.dkr.ecr.us-west-2.amazonaws.com"

aws ecr get-login-password --region us-west-2 | \
  docker login --password-stdin --username AWS $ECR_URL

docker tag sagemaker-build $ECR_URL/sagemaker_hls:latest

docker push $ECR_URL/sagemaker_hls:latest