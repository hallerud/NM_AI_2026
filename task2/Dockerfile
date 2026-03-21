# Dockerfile
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Command to run the application using uvicorn
# Cloud Run will set the PORT environment variable
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
