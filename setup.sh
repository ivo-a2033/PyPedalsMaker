# setup.sh — project-local setup

# Ensure script is sourced, not executed directly in a subshell
if [ "$0" = "$BASH_SOURCE" ] 2>/dev/null || [ "$0" = "$ZSH_EVAL_CONTEXT" ] 2>/dev/null; then
    echo "Error: Please run this script with 'source ./setup.sh' or '. ./setup.sh'"
    return 1 2>/dev/null || exit 1
fi

# 1. Create venv if missing
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# 2. Activate in current shell
source .venv/bin/activate

# 3. Upgrade pip (prevents weird install errors on fresh venvs)
pip install --upgrade pip --quiet

# 4. Install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    pip install -r requirements.txt
elif [ -f "pyproject.toml" ]; then
    echo "Installing project from pyproject.toml..."
    pip install -e .
fi

echo "Environment setup complete and activated!"