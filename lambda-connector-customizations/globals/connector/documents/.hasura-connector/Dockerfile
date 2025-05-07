FROM ghcr.io/hasura/ndc-python-lambda:v0.2.0

# Install system dependencies for python-magic
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic-dev \
    file \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt to the container
COPY requirements.txt /functions/

# Set the working directory
WORKDIR /functions

# Install Python dependencies in a virtual environment
RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install -r requirements.txt

# Copy the rest of the application code
COPY ./ /functions

# Create the group and user
RUN adduser -u 1000 python

# Fix ownership of files for non-root usage
RUN chown -R python:python /functions

# Switch to the non-root python user
USER python

#CMD ["venv/bin/python", "startup.py", "--log-level", "DEBUG"]
