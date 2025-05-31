VENV := ".venv"
PYTHON := VENV + "/bin/python3"
PIP := VENV + "/bin/pip"

default: install lint

install:
    @echo "Building the project..."
    python3 -m venv {{VENV}}
    {{VENV}}/bin/pip install -e .[testing]
    @echo "Successfully built the project."

lint:
    {{VENV}}/bin/black .
    {{VENV}}/bin/flake8 .
    {{VENV}}/bin/mypy .

clean:
    rm -rf __pycache__/
    rm -rf .mypy_cache/
    rm -rf {{VENV}}
    rm -rf build/
    rm -rf .coverage
    rm -rf *.egg-info
