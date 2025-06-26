FROM public.ecr.aws/lambda/python:3.9

# Install requirements
COPY app/requirements.txt .
RUN pip install -r requirements.txt

# Copy your code and model into the container
COPY app/ ./app/

# Set the CMD to your handler (filename.function_name)
CMD ["app.lambda_handler.lambda_handler"]
