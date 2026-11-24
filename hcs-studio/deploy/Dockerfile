# HCS Studio server image (API and workers). Build from the repository root (it needs the
# lab_events package next to hcs-studio):
#   docker compose -f hcs-studio/deploy/docker-compose.yml build
# GPU workers use the same image: PyTorch's CUDA wheels carry the CUDA runtime, the NVIDIA
# Container Toolkit provides the driver.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY lab_events /src/lab_events
COPY hcs-studio /src/hcs-studio
RUN pip install --no-cache-dir /src/lab_events "/src/hcs-studio[server,segmentation]"

# Run as an unprivileged user; settings (data roots, server options) live in a volume.
RUN useradd --system --uid 10001 --home /data hcs && mkdir -p /data/settings && chown -R hcs /data
USER hcs
ENV HCS_STUDIO_HOME=/data/settings
WORKDIR /data
EXPOSE 8000
CMD ["hcs-studio", "serve", "--host", "0.0.0.0", "--port", "8000"]
