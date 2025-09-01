# Use slim Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app



# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port
EXPOSE 10000

# Run the Flask app
CMD ["python", "app.py"]
