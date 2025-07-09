# Use an official Python runtime as a parent image
FROM python:3

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Expose the port that the Flask app will run on
EXPOSE 5000

# Define environment variable to ensure Python output is sent directly to the terminal
ENV PYTHONUNBUFFERED=1
ENV PYTHON_VERSION=3.8.19

# Run the Flask app
CMD ["python", "app.py"]
