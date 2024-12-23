Development Guide
================

Environment Setup
---------------

The Ananke2 project uses Python's built-in venv module for dependency management.

Version Compatibility
~~~~~~~~~~~~~~~~~~
- Python: 3.12+ required
- pip: Latest version recommended
- Operating Systems: Windows, macOS, Linux supported

Prerequisites
~~~~~~~~~~~~
- Python 3.12+
- Docker and Docker Compose
- Python venv module

Installation Steps
~~~~~~~~~~~~~~~~

1. Create and activate virtual environment::

    python -m venv .venv
    source .venv/bin/activate  # On Unix/macOS
    # or
    .venv\Scripts\activate  # On Windows

2. Configure pip to use BFSU mirror::

    pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple

3. Install dependencies::

    pip install -r requirements.txt
    pip install -r requirements-dev.txt  # For development

Best Practices
~~~~~~~~~~~~
- Always activate the virtual environment before running any pip commands
- Use requirements-dev.txt for development dependencies
- Keep requirements.txt updated when adding new dependencies

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
