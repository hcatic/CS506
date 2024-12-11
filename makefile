# Python interpreter to use
PYTHON := python3
VENV := venv
BIN := $(VENV)/bin

# Directory containing Jupyter notebooks
NOTEBOOK_DIR := notebooks
NOTEBOOKS := AirbnbEDA.ipynb AirbnbPreprocessing.ipynb AirbnbModels.ipynb

# Determine operating system
OS := $(shell uname -s)

all: run

setup: system-deps $(VENV)/bin/activate requirements.txt

# Install system-level dependencies
system-deps:
ifeq ($(OS), Darwin)  # macOS
	@echo "Installing libomp for macOS..."
	brew install libomp || echo "libomp already installed"
else ifeq ($(OS), Linux)  # Linux
	@echo "Installing libgomp for Linux..."
	if [ -x "$(command -v apt-get)" ]; then \
		sudo apt-get update && sudo apt-get install -y libgomp1; \
	elif [ -x "$(command -v yum)" ]; then \
		sudo yum install -y libgomp; \
	else \
		echo "Unsupported Linux distribution. Install libgomp manually."; \
	fi
else  # Windows or unsupported OS
	@echo "Ensure Microsoft Visual C++ Redistributable is installed for Windows."
endif

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

requirements.txt: $(VENV)/bin/activate
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install jupyter pytest

# Data processing target that runs notebooks in sequence
data: setup
	$(foreach notebook, $(NOTEBOOKS), \
		$(BIN)/jupyter nbconvert --to=script $(NOTEBOOK_DIR)/$(notebook); \
		$(BIN)/python $(NOTEBOOK_DIR)/$(basename $(notebook)).py;)

# Updated run target to use modeling.py instead of main.py
run: setup
	$(BIN)/python modeling.py

# Test target to run test_functions.py
test: setup
	$(BIN)/pytest test_functions.py

clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type f -name '.ipynb_checkpoints' -delete
	find $(NOTEBOOK_DIR) -type f -name '*.py' -delete