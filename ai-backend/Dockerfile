# Use Python 3.9 slim as base image
FROM python:3.9-slim

# Install OpenGL dependencies for OpenCV (specifically libGL.so.1)
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get clean

# Set the working directory
WORKDIR /app

# Copy all files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the backend port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
