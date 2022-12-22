# Use Python3.9
FROM python:3.9

# Create a directory to hold the application code inside the image
WORKDIR /usr/src/app

# Install module dependencies
# ADD requirements.txt ./
ADD requirements-cpu.txt ./
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements-cpu.txt

# Expose microservice port (6000)
EXPOSE 6000