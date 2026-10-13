# Smart Buildings 2026 — reproducible simulator environment.
#
# The simulator is pure NumPy with a Gymnasium environment API, so it installs
# cleanly on Python 3.12 with no conda/venv gymnastics.
FROM python:3.12-slim

# System libraries OpenCV needs at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip \
    && pip install -e .

# Default command: confirm the simulator imports end-to-end.
CMD ["python", "-c", "import smart_buildings.evaluate; print('smart_buildings simulator ready')"]
