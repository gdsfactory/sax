dev:
  uv venv --python 3.12 --clear
  uv sync --dev
  uv pip install -e .
  uv run pre-commit install

# bump version
bump version="patch":
    uvx --python 3.12 --from 'bver-cli==0.1.6' bver bump "{{ version }}"

dist:
  uv run python -m build --wheel

uv:
  curl -LsSf https://astral.sh/uv/install.sh | sh

ipykernel:
  uv run --dev python -m ipykernel install --user --name sax --display-name sax

# Fast checks without dependency synchronization or optional pytest plugins.
# Reuse compiled JAX executables between iterations; the first run fills the cache.
smoke:
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  JAX_COMPILATION_CACHE_DIR=.venv/.sax-smoke-jax-cache \
  JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
  JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0 \
  uv run --no-sync pytest src/tests/test_smoke.py -q

test: ipykernel
  uv run --dev pytest -s -n logical

cov: ipykernel
  uv run --dev pytest --cov=sax --cov-report=term-missing:skip-covered

docs:
  sed 's|](docs/|](|g' README.md > docs/index.md
  sed 's|^#|###|g' CHANGELOG.md | sed 's|^### \[0|## [0|g' > docs/changelog.md
  uv run zensical build

serve:
  uv run zensical serve

nbrun: ipykernel
  find nbs -maxdepth 2 -mindepth 1 -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*" -not -path "./.venv/*" | xargs parallel -j `nproc --all` uv run papermill {} {} -k sax :::

_nbdocs:
  rm -rf docs/nbs/examples
  rm -rf docs/nbs/internals
  mkdir -p docs/nbs/examples
  mkdir -p docs/nbs/internals
  find nbs/internals -maxdepth 1 -mindepth 1 -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*" -not -path "./.venv/*" | xargs parallel -j `nproc --all` uv run jupyter nbconvert --to markdown --embed-images {} --output-dir docs/nbs/internals ':::'
  find nbs/examples -maxdepth 1 -mindepth 1 -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*" -not -path "./.venv/*" | xargs parallel -j `nproc --all` uv run jupyter nbconvert --to markdown --embed-images {} --output-dir docs/nbs/examples ':::'

nbdocs: _nbdocs

nbclean-all:
  find . -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*" -not -path "./.venv/*" | xargs just nbclean

nbclean +filenames:
  for filename in {{filenames}}; do \
    uv run --no-sync nbstripout "$filename"; \
    uv run --no-sync nb-clean clean --remove-empty-cells "$filename"; \
    uv run --no-sync jq --indent 1 'del(.metadata.papermill)' "$filename" > "$filename.tmp" && mv "$filename.tmp" "$filename"; \
  done

tree:
  @tree -a -I .git --gitignore

clean: nbclean-all
  rm -rf .venv
  rm -rf docs/nbs/*
  rm -rf site
  rm -rf dist
  find . -name "*.egg_info" | xargs rm -rf
  find . -name "*.so" | xargs rm -rf
  find . -name ".ipynb_checkpoints" | xargs rm -rf
  find . -name ".virtual_documents" | xargs rm -rf
  find . -name ".mypy_cache" | xargs rm -rf
  find . -name ".pytest_cache" | xargs rm -rf
  find . -name ".ruff_cache" | xargs rm -rf
  find . -name "__pycache__" | xargs rm -rf
  find . -name "build" | xargs rm -rf
  find . -name "builds" | xargs rm -rf
  find . -name "modes" | xargs rm -rf
  find src -name "*.c" | xargs rm -rf
  find src -name "*.pyc" | xargs rm -rf
  find src -name "*.pyd" | xargs rm -rf
  find src -name "*.so" | xargs rm -rf
