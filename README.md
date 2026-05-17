# devintel-engine

Worker for analyzing a repository in an isolated process. It uses [devaudt](https://github.com/jhetjhet/devaudt) for deterministic extraction and an LLM for higher-level analysis.

By default, LLM is enabled (premium mode). You can disable it for minimum-service mode while keeping the same output structure:

```sh
DEVINTEL_WITH_LLM=0 docker compose run --rm engine \
	python3 /app/orchestrator.py <repo_url> <your_id>
```

Alias: `DEVINTEL_LLM_ENABLED` also works.

## Build

```sh
docker compose build
```

## Run

```sh
docker compose run --rm engine python3 /app/orchestrator.py <repo_url> <your_id>
```

Or run the image directly:

```sh
docker run --rm <engine_image_name> python3 /app/orchestrator.py <repo_url> <your_id>
```