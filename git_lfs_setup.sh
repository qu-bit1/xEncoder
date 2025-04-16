#!/bin/bash

# Install git-lfs if not already installed
if ! command -v git-lfs &> /dev/null; then
    echo "Git LFS not found. Installing..."
    # For macOS with homebrew
    if command -v brew &> /dev/null; then
        brew install git-lfs
    # For Ubuntu/Debian
    elif command -v apt-get &> /dev/null; then
        sudo apt-get install git-lfs
    else
        echo "Please install git-lfs manually: https://git-lfs.github.com/"
        exit 1
    fi
fi

# Initialize git-lfs
git lfs install

# Track large file types
git lfs track "*.h5"
git lfs track "*.h5ad"
git lfs track "*.parquet"
git lfs track "*.npz"
git lfs track "*.npy"
git lfs track "*.pt"

# Add .gitattributes to repository
git add .gitattributes

echo "Git LFS setup complete. Large files will now be tracked properly."
echo "Use regular git commands to add and commit your files:"
echo "git add ."
echo "git commit -m 'Your commit message'"
echo "git push"