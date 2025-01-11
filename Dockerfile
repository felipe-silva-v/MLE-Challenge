# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy dependencies and install them
COPY requirements.txt .
RUN pip install --default-timeout=3600 --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# Copy the source code
COPY . .

# Expose the port used by the API
EXPOSE 8080

# Command to start the API with the dynamic port
CMD ["sh", "-c", "uvicorn challenge.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
