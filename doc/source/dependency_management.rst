Dependency Management
===================

Overview
--------
The Ananke2 project uses Python's built-in venv module for dependency management, with pip configured to use the BFSU mirror for better accessibility in Chinese development environments.

Key Files
---------
- requirements.txt: Core project dependencies
- requirements-dev.txt: Additional development dependencies

Configuration
------------
The project uses the BFSU mirror for pip package installation::

    pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple

This configuration is automatically set in the Dockerfile for containerized environments.

Adding Dependencies
-----------------
1. Add new packages to the appropriate requirements file
2. Document version constraints if needed
3. Test installation in a clean virtual environment
4. Update documentation if the new package requires additional setup

Development Workflow
------------------
1. Always work within the virtual environment
2. Update requirements files when adding/removing packages
3. Test changes in a clean virtual environment before committing
