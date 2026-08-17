.PHONY: start stop rebuild start-debug start-native stop-native generate-data setup test logs logs-mcp shell start-gpt stop-gpt stop-ngrok

UNAME := $(shell uname)

# =============================================================================
# Docker Compose commands
# =============================================================================

# Start the full application stack using Docker Compose
# Web app runs in a container with local code bind-mounted (hot reload)
# Prerequisites: Docker and Docker Compose must be installed
# Usage: make start
start:
	@echo "Starting Anistroph application..."
	@echo "Detected OS: $(UNAME)"

	# Start Docker Desktop on macOS if not running
	@if [ "$(UNAME)" = "Darwin" ]; then \
		echo "Opening Docker Desktop GUI on Mac."; \
		open -a Docker || open -a "Docker Desktop"; \
	else \
		echo "No GUI launch command for non-Darwin OS."; \
	fi

	# Wait for Docker daemon
	@echo "Waiting for Docker daemon to become available..."
	@TIMEOUT=30; \
	while ! docker info >/dev/null 2>&1; do \
		if [ $$TIMEOUT -le 0 ]; then \
			echo "Error: Docker daemon did not start within 30 seconds. Please check Docker Desktop."; \
			exit 1; \
		fi; \
		printf '.'; \
		sleep 1; \
		TIMEOUT=$$(($$TIMEOUT - 1)); \
	done; \
	echo ""; \
	echo "Docker daemon is running. Proceeding with compose."

	# Start the services
	@if ! docker compose up -d; then \
		echo "Error: Failed to start services with docker compose"; \
		exit 1; \
	fi

	@echo ""
	@echo "Anistroph application started successfully."
	@echo "  Web UI:     http://localhost:9500"
	@echo "  API docs:   http://localhost:9500/docs"
	@echo "  MCP (HTTP): http://localhost:9500/mcp"
	@echo "  OpenAPI:    http://localhost:9500/openapi.json"
	@echo "  Health:     http://localhost:9500/health"

# Stop all Docker containers
stop:
	@echo "Stopping Anistroph application..."
	docker compose down
	@echo "Anistroph application stopped successfully."

# Rebuild and start with latest code changes (rebuilds Docker image)
rebuild:
	@echo "Rebuilding and starting Anistroph..."
	docker compose up -d --build
	@echo "Anistroph rebuilt and started. Access at http://localhost:9500"

# =============================================================================
# Native (non-Docker) commands — uses local .venv
# =============================================================================

# Run FastAPI server in foreground with live reload (local .venv)
start-native:
	@echo "Starting Anistroph in native mode (local .venv)..."
	@. .venv/bin/activate && uvicorn backend.main:app --reload --host 0.0.0.0 --port 9500

# Stop native uvicorn (finds and kills the process on port 9500)
stop-native:
	@echo "Stopping native Anistroph..."
	@lsof -ti:9500 | xargs kill -9 2>/dev/null || echo "No process on port 9500"
	@echo "Stopped."

# Run in debug mode with verbose logging
start-debug:
	@echo "Starting Anistroph in debug mode..."
	@. .venv/bin/activate && uvicorn backend.main:app --reload --host 0.0.0.0 --port 9500 --log-level debug

# =============================================================================
# Data generation
# =============================================================================

# Generate the full synthetic predictive-maintenance dataset
# Usage: make generate-data
# Custom: make generate-data MACHINES=50 DAYS=60 INTERVAL=5 SEED=42
generate-data:
	@echo "Generating synthetic predictive-maintenance data..."
	@. .venv/bin/activate && python scripts/generate_sensor_data.py \
		--machines $(MACHINES) --days $(DAYS) --interval $(INTERVAL) --seed $(SEED)

# =============================================================================
# One-shot setup: generate all synthetic data + register all reference datasets
# =============================================================================

# Generate all three synthetic datasets and register all eleven dataset configs.
# Idempotent — skips generation/registration for datasets already present.
# Usage: make setup
# Force re-registration: make setup SETUP_ARGS=--force
# Skip generation (data already on disk): make setup SETUP_ARGS=--skip-gen
setup:
	@echo "Setting up Anistroph reference datasets..."
	@. .venv/bin/activate && python scripts/setup_datasets.py $(SETUP_ARGS)
	@echo ""
	@echo "Setup complete. Start the server with:"
	@echo "  make start         (Docker Compose, port 9500)"
	@echo "  make start-native  (local .venv, port 9500)"

# =============================================================================
# Testing
# =============================================================================

# Run the full test suite
test:
	@echo "Running Anistroph test suite..."
	@. .venv/bin/activate && pytest -v

# =============================================================================
# Logs
# =============================================================================

# Stream webapp logs live
logs:
	docker compose logs -f webapp

# Stream MCP server logs live
logs-mcp:
	docker compose logs -f mcp

# =============================================================================
# Utility
# =============================================================================

# Open a shell inside the running webapp container
shell:
	docker compose exec webapp bash

# =============================================================================
# ChatGPT GPT Action (ngrok tunnel)
# =============================================================================

# Load NGROK_AUTHTOKEN from .env if present
-include .env
export NGROK_AUTHTOKEN

# Start the native server + ngrok tunnel for ChatGPT GPT Actions.
# Displays the public URL and the filtered OpenAPI spec URL.
# Requires NGROK_AUTHTOKEN in .env or environment.
# Usage: make start-gpt
start-gpt:
	@echo "Starting Anistroph for ChatGPT GPT Actions..."
	@echo ""
	# Verify ngrok authtoken is available
	@if [ -z "$$NGROK_AUTHTOKEN" ]; then \
		echo "Error: NGROK_AUTHTOKEN is not set."; \
		echo "Add it to .env:  echo 'NGROK_AUTHTOKEN=<your-token>' >> .env"; \
		echo "Get a free token at https://dashboard.ngrok.com/get-started/your-authtoken"; \
		exit 1; \
	fi
	@echo "Using NGROK_AUTHTOKEN from .env"
	# Kill any existing process on port 9500
	@lsof -ti:9500 | xargs kill -9 2>/dev/null || true
	# Kill any existing ngrok tunnel
	@pkill -f "ngrok http 9500" 2>/dev/null || true
	# Start the native server in the background
	@. .venv/bin/activate && nohup uvicorn backend.main:app --host 0.0.0.0 --port 9500 > /tmp/anistroph_server.log 2>&1 &
	@echo "Waiting for server to start..."
	@sleep 3
	@if ! curl -s http://localhost:9500/health >/dev/null 2>&1; then \
		echo "Error: server did not start. Check /tmp/anistroph_server.log"; \
		exit 1; \
	fi
	@echo "Server is running on http://localhost:9500"
	# Start ngrok tunnel in the background (authtoken passed via env var)
	@nohup ngrok http 9500 > /tmp/anistroph_ngrok.log 2>&1 &
	@echo "Waiting for ngrok tunnel..."
	@sleep 3
	# Fetch the public URL from ngrok's local API
	@NGROK_URL=$$(curl -s http://localhost:4040/api/tunnels | .venv/bin/python -c "import sys,json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])" 2>/dev/null); \
	if [ -z "$$NGROK_URL" ]; then \
		echo "Error: ngrok tunnel did not start. Check /tmp/anistroph_ngrok.log"; \
		exit 1; \
	fi; \
	echo ""; \
	echo "========================================================"; \
	echo "  Anistroph is now public for ChatGPT GPT Actions"; \
	echo "========================================================"; \
	echo ""; \
	echo "  Public URL:        $$NGROK_URL"; \
	echo "  MCP (HTTP):        $$NGROK_URL/mcp"; \
	echo "  OpenAPI (GPT):     $$NGROK_URL/openapi-gpt.json"; \
	echo "  OpenAPI (full):    $$NGROK_URL/openapi.json"; \
	echo "  Health:            $$NGROK_URL/health"; \
	echo ""; \
	echo "  To configure ChatGPT:"; \
	echo "    1. Go to https://chat.openai.com/gpts"; \
	echo "    2. Create a new GPT -> Configure -> Actions"; \
	echo "    3. Import from URL: $$NGROK_URL/openapi-gpt.json"; \
	echo "    4. No auth required"; \
	echo ""; \
	echo "  To stop:  make stop-gpt"; \
	echo "========================================================"; \
	echo ""; \
	echo "  Server log:  /tmp/anistroph_server.log"; \
	echo "  Ngrok log:   /tmp/anistroph_ngrok.log"; \
	echo "  Ngrok dashboard: http://localhost:4040"

# Stop both the ngrok tunnel and the native server.
# Usage: make stop-gpt
stop-gpt:
	@echo "Stopping Anistroph GPT Action tunnel..."
	@pkill -f "ngrok http 9500" 2>/dev/null && echo "  ngrok tunnel stopped." || echo "  ngrok was not running."
	@lsof -ti:9500 | xargs kill -9 2>/dev/null && echo "  server stopped." || echo "  server was not running."
	@echo "Anistroph GPT Action stopped. The public URL is no longer accessible."

# Stop only the ngrok tunnel (keep the local server running).
# Usage: make stop-ngrok
stop-ngrok:
	@echo "Stopping ngrok tunnel only (server stays on localhost:9500)..."
	@pkill -f "ngrok http 9500" 2>/dev/null && echo "  ngrok tunnel stopped. Public URL no longer accessible." || echo "  ngrok was not running."
