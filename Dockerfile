FROM python:3.11-slim

LABEL maintainer="Rushikesh"
LABEL description="Monolithic Training vs Modular DVC Pipeline"

# Install system-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
        bash \
        sed \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Configure git globally — required for dvc exp commands to create commits
RUN git config --global user.email "rushi5706@github.com" \
 && git config --global user.name "Rushikesh" \
 && git config --global --add safe.directory /app \
 && git config --global init.defaultBranch main

# Copy dependency specification first to leverage layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project source files (excludes items listed in .dockerignore)
COPY . .

# Initialize a fresh Git repository inside the image
# (We do not ship the host .git directory)
RUN git init \
 && git config user.email "rushi5706@github.com" \
 && git config user.name "Rushikesh"

# Initialize DVC with Git integration
RUN dvc init

# Set up the local DVC remote storage path
RUN mkdir -p /root/dvc-remote \
 && dvc remote add -d storage /root/dvc-remote

# Create required output directories
RUN mkdir -p data models metrics

# Download the UCI Adult Income dataset at build time
RUN curl -fsSL \
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data" \
    -o /tmp/adult.data \
 && printf 'age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income\n' \
    > data/adult.csv \
 && cat /tmp/adult.data >> data/adult.csv \
 && rm /tmp/adult.data

# Track raw data with DVC and make an initial Git commit
RUN dvc add data/adult.csv \
 && git add -A \
 && git commit -m "initial: project setup with data and DVC pipeline"

# Run the full DVC pipeline — populates .dvc/cache and output artifacts
RUN dvc repro

# Push artifacts to local remote
RUN dvc push

# Commit pipeline lock file and DVC state
RUN git add -A \
 && git commit -m "feat: run baseline DVC pipeline and cache all stage outputs"

# Healthcheck — verifies Python environment and key libraries are available
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sklearn; import dvc; import pandas; print('healthy')" || exit 1

# Keep container running — all actual work is done via docker-compose run
CMD ["tail", "-f", "/dev/null"]
