#!/bin/bash

APP_NAME="rentang-streamlit"
IMAGE_NAME="rentang-streamlit-image"

# Build image
echo "🔨 Building Docker image..."
docker build -t $IMAGE_NAME .

# Check if container already exists
if [ "$(docker ps -aq -f name=^${APP_NAME}$)" ]; then
  echo "🛑 Stopping & removing old container..."
  docker stop $APP_NAME >/dev/null 2>&1
  docker rm $APP_NAME >/dev/null 2>&1
fi

# Run new container
echo "🚀 Running new container..."
docker run -d --name $APP_NAME -p 8501:8501 $IMAGE_NAME
