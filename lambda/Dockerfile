FROM public.ecr.aws/lambda/python:3.9

# Set working directory
WORKDIR /var/task

# Upgrade pip
RUN pip install --upgrade pip

# Pre-install compatible NumPy version
RUN pip install "numpy<2.0"

# Copy and install remaining dependencies
COPY app/requirements.txt .
RUN pip install -r requirements.txt

# Copy the application code
COPY app/ ./app/

# Set Lambda entrypoint
CMD ["app.lambda_handler.lambda_handler"]
