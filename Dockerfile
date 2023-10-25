# Use the official Python image as the base image
FROM python:3.9

# 以下の処理を行っていくときのuser
USER root

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --trusted-host pypi.python.org -r requirements.txt

RUN apt-get update && apt-get install -y \
    libzbar0 \
    libgl1-mesa-glx\
    libsndfile1

EXPOSE 80 80

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy the rest of the application code
COPY . .

