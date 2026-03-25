FROM python:3.11-slim

WORKDIR /app

# Install the package
COPY pyproject.toml .
COPY alignscope/ alignscope/
COPY frontend/ frontend/
COPY README.md .

RUN pip install --no-cache-dir .

# Expose dashboard port
EXPOSE 8000

# Default: start in demo mode
CMD ["alignscope", "start", "--demo", "--port", "8000"]
