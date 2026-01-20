.PHONY: install ingest index run test test-integration eval docker-up docker-down docker-build

install:
	pip install -r requirements.txt

ingest:
	python -m app.rag.ingest

index:
	python -m app.rag.index

run:
	uvicorn app.main:app --reload

test:
	pytest -q

test-integration:
	pytest -q -m integration

eval:
	python -m eval.run_eval

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down