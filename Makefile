.PHONY: start stop rebuild start-debug start-native stop-native generate-data test logs logs-mcp shell

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
