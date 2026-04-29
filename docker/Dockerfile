# syntax=docker/dockerfile:1.7-labs

# ---------- Stage 1: build the React frontend ----------
FROM node:20-alpine AS frontend
WORKDIR /app/webapp/frontend

COPY webapp/frontend/package.json webapp/frontend/package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci --no-audit --no-fund; else npm install --no-audit --no-fund; fi

COPY webapp/frontend/ ./
RUN npm run build


# ---------- Stage 2: Python runtime with CUDA ----------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/usr/local/bin:${PATH}" \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3-pip \
        curl \
        ca-certificates \
        git \
        tzdata \
        libgl1 \
        libglib2.0-0 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python -m pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-web.txt

COPY trading_system/ /app/trading_system/
COPY webapp/ /app/webapp/
COPY config/ /app/config/
COPY pyproject.toml run.sh /app/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

# Drop the dev source for the frontend; copy the built dist into place
RUN rm -rf /app/webapp/frontend/node_modules /app/webapp/frontend/src
COPY --from=frontend /app/webapp/frontend/dist /app/webapp/frontend/dist

RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 28473

ENV BOARDROOM_TB_UPSTREAM=http://tensorboard:6006

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "webapp.backend.main:app", "--host", "0.0.0.0", "--port", "28473"]
