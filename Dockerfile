# Start with a lightweight Linux Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements first (to cache the heavy pip installs)
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application (Code and Models)
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the production web server
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]