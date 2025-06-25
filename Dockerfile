# Use Python 3.10 slim image
FROM python:3.10-slim

# Set environment variables to prevent .pyc files and buffer issues
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && apt-get clean

# Copy all project files into the container
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Expose the port your app runs on (adjust if needed)
EXPOSE 10000

# Run the FastAPI app
CMD ["uvicorn", "fastapi:app", "--host", "0.0.0.0", "--port", "10000"]


