# Use Python 3.9 as the base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy all files to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create the uploads folder for storing images
RUN mkdir -p /app/uploads

# Expose the port that the UI backend will run on
EXPOSE 8000

# Run the UI backend
CMD ["python", "app.py"]
