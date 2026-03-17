FROM python:3.11-slim

LABEL maintainer="SAR Generator Team"
LABEL description="Synthetic SAR Log Data Generator for Telco Cloud"

WORKDIR /app

# System dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Default output directory
RUN mkdir -p /app/output

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "from config import get_default_config; print('OK')" || exit 1

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]

# Usage examples:
#   docker build -t sar-generator .
#   docker run -v $(pwd)/output:/app/output sar-generator
#   docker run -v $(pwd)/output:/app/output sar-generator -c config.yaml
#   docker run -v $(pwd)/output:/app/output sar-generator --write-example-config
