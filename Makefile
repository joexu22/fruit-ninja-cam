.PHONY: setup run test download-models preview

setup:
	python3 -m pip install -e ".[dev]"
	python3 scripts/download_models.py

download-models:
	python3 scripts/download_models.py

run:
	python3 -m fruit_ninja_cam

test:
	python3 -m pytest -q

preview:
	python3 scripts/render_preview.py --sheet
	python3 scripts/render_preview.py --hero
	python3 scripts/render_preview.py
