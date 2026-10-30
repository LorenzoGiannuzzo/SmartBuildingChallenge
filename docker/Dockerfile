# Environment for the experiments (simulator, PyTorch on CPU, plotting).
#
# The image holds only the environment: the repository is mounted at /work by
# docker-compose.yml, so a git pull updates the code without a rebuild, and
# data, models and results stay on the server disk.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu

ENV PYTHONPATH=/work MPLCONFIGDIR=/tmp/matplotlib PYTHONUNBUFFERED=1
WORKDIR /work
CMD ["python", "-c", "import torch, smart_buildings.evaluate, sbwm.policy; print('sbwm environment ready, torch', torch.__version__)"]
