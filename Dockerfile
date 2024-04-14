# Use an official Python runtime as a parent image
FROM python:3.11.7-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install any needed packages specified in requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code
COPY . .

# Make port available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV PORT=5000

# Run app.py when the container launches
CMD gunicorn -b 0.0.0.0:$PORT app:app
