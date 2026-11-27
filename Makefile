.PHONY: experiment api dashboard test lint docker

PYTHON ?= python

experiment:
	$(PYTHON) scripts/run_drift_experiments.py

api:
	$(PYTHON) -m uvicorn api.app:app --host 127.0.0.1 --port 8030

dashboard:
	$(PYTHON) -m streamlit run streamlit_app/app.py

lint:
	$(PYTHON) -m ruff check .

test: lint
	$(PYTHON) -m pytest -q

docker:
	docker compose up --build
