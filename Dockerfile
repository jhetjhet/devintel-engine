FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    curl git npm cloc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install semgrep redis \
    langgraph \
    langchain-openai \
    pydantic

WORKDIR /app

# build from https://github.com/jhetjhet/devaudt.git
COPY ./devaudt-0.1.0-py3-none-any.whl /tmp/devaudt-0.1.0-py3-none-any.whl

RUN pip install /tmp/devaudt-0.1.0-py3-none-any.whl && rm /tmp/devaudt-0.1.0-py3-none-any.whl

# Copy application source
COPY src/ /app/

WORKDIR /app

CMD ["python", "orchestrator.py"]