# Frozen study environment (plan section 3, "Docker for the frozen study").
# The model is NOT in this image: Ollama runs on the host and the container
# reaches it through LLM_BASE_URL.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LLM_BASE_URL=http://host.docker.internal:11434/v1

# git lets each run log record the exact commit it came from.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt

COPY . .
RUN git config --global --add safe.directory /app

CMD ["python", "-m", "src.sim.run", "--config", "configs/small.yaml"]
