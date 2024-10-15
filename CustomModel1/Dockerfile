FROM python:3.9-slim

# Install required libraries including Flask
RUN pip install --upgrade pip && \
    pip install transformers torch flask

# Copy your application and model script
COPY . /app
WORKDIR /app

# Load the Hugging Face model (optional step if you want to pre-load the model)
RUN python -c "from transformers import pipeline; classifier = pipeline('text-classification', model='SamLowe/roberta-base-go_emotions'); classifier.save_pretrained('./model')"

# Define the entrypoint for the container
ENTRYPOINT ["python", "serve.py"]