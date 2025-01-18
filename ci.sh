#!/bin/bash

set -ex -o pipefail

# Log some general info about the environment
echo "::group::Environment"
uname -a
env | sort
PROJECT='azul'
echo "::endgroup::"


################################################################
# We have a Python environment!
################################################################

echo "::group::Versions"
python -c "import sys, struct; print('python:', sys.version); print('version_info:', sys.version_info); print('bits:', struct.calcsize('P') * 8)"
echo "::endgroup::"

echo "::group::Install dependencies"
python -m pip install -U pip tomli
python -m pip --version
UV_VERSION=$(python -c 'import tomli; from pathlib import Path; print({p["name"]:p for p in tomli.loads(Path("uv.lock").read_text())["package"]}["uv"]["version"])')
python -m pip install uv==$UV_VERSION
python -m uv --version

UV_VENV_SEED="pip"
UV_VENV_OUTPUT="$(uv venv --seed --allow-existing 2>&1)"
echo "$UV_VENV_OUTPUT"

# Extract the activation command from the output
activation_command=$(echo "$UV_VENV_OUTPUT" | grep -oP '(?<=Activate with: ).*')

# Check if the activation command was found
if [ -n "$activation_command" ]; then
    # Execute the activation command
    echo "Activating virtual environment..."
    eval "$activation_command"
else
    echo "::error:: Activation command not found in uv venv output."
    exit 1
fi
python -m pip install uv==$UV_VERSION

# python -m uv build
# wheel_package=$(ls dist/*.whl)
# python -m uv pip install "$PROJECT @ $wheel_package" -c test-requirements.txt

if [ "$CHECK_FORMATTING" = "1" ]; then
    python -m uv sync --extra tests --extra tools
    echo "::endgroup::"
    source check.sh
else
    # Actual tests
    # expands to 0 != 1 if NO_TEST_REQUIREMENTS is not set, if set the `-0` has no effect
    # https://pubs.opengroup.org/onlinepubs/9699919799/utilities/V3_chap02.html#tag_18_06_02
    if [ "${NO_TEST_REQUIREMENTS-0}" == 1 ]; then
        # python -m uv pip install pytest coverage -c test-requirements.txt
        python -m uv sync --extra tests
        flags=""
        #"--skip-optional-imports"
    else
        python -m uv sync --extra tests --extra tools
        flags=""
    fi

    echo "::endgroup::"

    echo "::group::Setup for tests"

    # We run the tests from inside an empty directory, to make sure Python
    # doesn't pick up any .py files from our working dir. Might have been
    # pre-created by some of the code above.
    mkdir empty || true
    cd empty

    INSTALLDIR=$(python -c "import os, $PROJECT; print(os.path.dirname($PROJECT.__file__))")
    cp ../pyproject.toml "$INSTALLDIR"

    # get mypy tests a nice cache
    MYPYPATH=".." mypy --config-file= --cache-dir=./.mypy_cache -c "import $PROJECT" >/dev/null 2>/dev/null || true

    # support subprocess spawning with coverage.py
    # echo "import coverage; coverage.process_startup()" | tee -a "$INSTALLDIR/../sitecustomize.py"

    echo "::endgroup::"
    echo "::group:: Run Tests"
    if coverage run --rcfile=../pyproject.toml -m pytest -ra --junitxml=../test-results.xml ../tests --verbose --durations=10 $flags; then
        PASSED=true
    else
        PASSED=false
    fi
    PREV_DIR="$PWD"
    cd "$INSTALLDIR"
    rm pyproject.toml
    cd "$PREV_DIR"
    echo "::endgroup::"
    echo "::group::Coverage"

    coverage combine --rcfile ../pyproject.toml
    coverage report -m --rcfile ../pyproject.toml
    coverage xml --rcfile ../pyproject.toml

    echo "::endgroup::"
    $PASSED
fi
