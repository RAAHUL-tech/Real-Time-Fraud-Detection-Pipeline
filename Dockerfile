# Use an official Python base image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Copy requirements.txt first for caching layer
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project code into the container
COPY . .

# Default command to run your training script (can be overridden)
CMD ["python", "src/train.py"]
