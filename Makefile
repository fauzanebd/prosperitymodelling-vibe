.PHONY: dev db-up db-down db-reset train-models

# Development server
dev:
	./run_dev.sh

# Database operations
db-up:
	docker-compose -f docker-compose.dev.yml up -d

db-down:
	docker-compose -f docker-compose.dev.yml down

db-reset:
	docker-compose -f docker-compose.dev.yml down -v
	docker-compose -f docker-compose.dev.yml up -d

# Model operations
train-models:
	python train_models.py

# Help
help:
	@echo "Available commands:"
	@echo "  make dev          - Run development server"
	@echo "  make db-up        - Start database container"
	@echo "  make db-down      - Stop database container"
	@echo "  make db-reset     - Reset database (delete all data and recreate)"
	@echo "  make train-models - Train machine learning models" 