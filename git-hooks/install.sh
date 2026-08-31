#!/usr/bin/env sh
# Install the tracked git-hooks into the local git config.
# Run once after cloning: sh git-hooks/install.sh
set -e

ROOT="$(git rev-parse --show-toplevel)"
git config core.hooksPath "$ROOT/git-hooks"
chmod +x "$ROOT/git-hooks/"*
echo "✓ git hooks installed (core.hooksPath = git-hooks/)"
