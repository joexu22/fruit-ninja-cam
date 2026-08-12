.PHONY: setup run test download-models

setup:
	python3 -m pip install -e ".[dev]"
	python3 scripts/download_models.py

download-models:
	python3 scripts/download_models.py

run:
	python3 -m fruit_ninja_cam

test:
	python3 -m pytest -q
