FROM python:3.11-slim

LABEL maintainer="Rushikesh"
LABEL description="Monolithic Training vs Modular DVC Pipeline"

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
        bash \
        sed \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git config --global user.email "rushi5706@github.com" \
 && git config --global user.name "Rushikesh" \
 && git config --global --add safe.directory /app \
 && git config --global init.defaultBranch main

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN git init \
 && git config user.email "rushi5706@github.com" \
 && git config user.name "Rushikesh"

RUN rm -rf .dvc && dvc init

RUN mkdir -p /root/dvc-remote \
 && dvc remote add -d storage /root/dvc-remote

RUN mkdir -p data models metrics

RUN curl -fsSL \
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data" \
    -o /tmp/adult.data \
 && printf 'age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income\n' \
    > data/adult.csv \
 && cat /tmp/adult.data >> data/adult.csv \
 && rm /tmp/adult.data

RUN dvc add data/adult.csv \
 && git add -A \
 && git commit -m "initial: project setup with DVC pipeline and dataset"

RUN dvc repro

RUN dvc push

RUN git add -A \
 && git commit -m "feat: baseline DVC pipeline executed and cached"

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sklearn; import dvc; import pandas; print('healthy')" || exit 1

CMD ["tail", "-f", "/dev/null"]
