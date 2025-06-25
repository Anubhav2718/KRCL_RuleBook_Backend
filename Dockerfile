# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the FastAPI app
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "fastapi:app", "--host", "0.0.0.0", "--port", "8000"]

