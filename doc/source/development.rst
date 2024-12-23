Development Guide
================

Environment Setup
---------------

The Ananke2 project uses Python's built-in venv module for dependency management.

Prerequisites
~~~~~~~~~~~~
- Python 3.12+
- Docker and Docker Compose
- Python venv module

Installation Steps
~~~~~~~~~~~~~~~~

1. Clone the repository::

    git clone https://github.com/ceresman/ananke2.git
    cd ananke2

2. Create and activate virtual environment::

    python -m venv .venv
    source .venv/bin/activate  # On Unix/macOS
    # or
    .venv\Scripts\activate  # On Windows

3. Configure pip to use BFSU mirror::

    pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple

4. Install dependencies::

    pip install -r requirements.txt
    pip install -r requirements-dev.txt  # For development

5. Configure environment::

    cp .env.example .env
    # Edit .env with your credentials and API keys

6. Start services::

    docker-compose up -d

Development Workflow
------------------

Running Tests
~~~~~~~~~~~~
To run the test suite::

    python -m pytest tests/

Code Style
~~~~~~~~~
We use the following tools for code formatting and linting:

- black: Code formatting
- isort: Import sorting
- flake8: Style guide enforcement

To format and lint your code::

    black .
    isort .
    flake8
