#!/usr/bin/env bash
# setup.sh - assemble the CUDA-compat PyTorch-for-SAIL source tree.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PATCH_DIR="$REPO_ROOT/patches"

# Keep the assembled sources under the deployment root; WORK overrides the parent.
WORK="${WORK:-$REPO_ROOT}"
UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/t-head/pytorch-for-sail.git}"
UPSTREAM_BRANCH="${UPSTREAM_BRANCH:-v2.14.1}"
REFERENCE_REPO="${REFERENCE_REPO:-}"

# A reference repository supplies object caches, never the upstream revision.
CLONE_REFERENCE_ARGS=()
if [ -n "$REFERENCE_REPO" ]; then
    REFERENCE_REPO="$(git -C "$REFERENCE_REPO" rev-parse --show-toplevel)"
    CLONE_REFERENCE_ARGS=(--reference-if-able "$REFERENCE_REPO" --dissociate)
fi

ensure_clean_checkout() {
    local state worktree_status
    for state in rebase-apply rebase-merge MERGE_HEAD CHERRY_PICK_HEAD REVERT_HEAD; do
        if [ -e "$(git rev-parse --git-path "$state")" ]; then
            echo "ERROR: unfinished Git operation ($state); resolve it before setup." >&2
            exit 1
        fi
    done
    worktree_status="$(git status --porcelain --untracked-files=normal --ignore-submodules="${1:-none}")"
    if [ -n "$worktree_status" ]; then
        echo "ERROR: pytorch worktree is not clean; preserve local changes before setup." >&2
        exit 1
    fi
}

mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
PYTORCH_DIR="$WORK/pytorch"

if [ -e "$PYTORCH_DIR/.git" ]; then
    echo "Reusing existing pytorch checkout at $PYTORCH_DIR"
    cd "$PYTORCH_DIR"
    ensure_clean_checkout
else
    echo "Cloning $UPSTREAM_URL branch $UPSTREAM_BRANCH into $PYTORCH_DIR"
    if ! git clone "${CLONE_REFERENCE_ARGS[@]}" -b "$UPSTREAM_BRANCH" \
            --single-branch "$UPSTREAM_URL" "$PYTORCH_DIR"; then
        echo "ERROR: upstream clone failed; existing files were not removed." >&2
        exit 1
    fi
    cd "$PYTORCH_DIR"
fi

# Fetch the explicit public upstream, even if an existing origin points elsewhere.
# Stop on failure instead of falling back to cached or local development commits.
echo "Fetching latest $UPSTREAM_URL branch $UPSTREAM_BRANCH"
if ! git fetch --no-tags --no-recurse-submodules "$UPSTREAM_URL" "refs/heads/$UPSTREAM_BRANCH"; then
    echo "ERROR: upstream fetch failed; refusing to use an older local commit." >&2
    exit 1
fi
UPSTREAM_HEAD="$(git rev-parse --verify 'FETCH_HEAD^{commit}')"
echo "Upstream base commit: $UPSTREAM_HEAD"

# Use only this fetch result as the checkout and patch base.
ensure_clean_checkout
if ! git checkout --detach "$UPSTREAM_HEAD"; then
    echo "ERROR: cannot check out latest upstream commit $UPSTREAM_HEAD" >&2
    exit 1
fi
# Submodules may still be at old gitlinks after switching the parent revision.
# Their local changes were checked above; synchronize their revisions below.
ensure_clean_checkout all

# Check all pure-diff patches before applying and staging; do not create commits.
echo "Applying compat patches from $PATCH_DIR"
if ! git apply --check --index "$PATCH_DIR"/*.patch; then
    echo "ERROR: compat patches do not apply to upstream base $UPSTREAM_HEAD; inspect the failure before retrying." >&2
    exit 1
fi
git apply --index "$PATCH_DIR"/*.patch
echo "Compat patches applied to the worktree and index; no commits were created."

# Initialize dependencies at the assembled gitlinks, not a reference worktree.
# Keep submodule gitdirs independent of any local reference repository.
echo "Syncing and updating submodules (this may take a while)"
git submodule sync --recursive
git submodule update --init --recursive

# Copy the optional Docker asset without modifying the upstream docker/sail tree.
# Keep it locally ignored; staged patches still need preserving before reassembly.
# The upstream .dockerignore points to .gitignore, which excludes build/ and dist/.
DOCKERFILE_SRC="$REPO_ROOT/docker/Dockerfile.sdk2.2.1-cuda-compat"
if [ -f "$DOCKERFILE_SRC" ]; then
    mkdir -p "$PYTORCH_DIR/docker"
    cp "$DOCKERFILE_SRC" "$PYTORCH_DIR/docker/"
    echo "Copied docker build asset into $PYTORCH_DIR/docker"
    EXCLUDE_FILE="$(git -C "$PYTORCH_DIR" rev-parse --git-path info/exclude)"
    grep -qxF '/docker/Dockerfile.sdk2.2.1-cuda-compat' "$EXCLUDE_FILE" 2>/dev/null \
        || echo '/docker/Dockerfile.sdk2.2.1-cuda-compat' >> "$EXCLUDE_FILE"
fi

echo ""
echo "Source tree ready: $PYTORCH_DIR"
echo "Upstream: $UPSTREAM_URL branch $UPSTREAM_BRANCH"
echo "Upstream base commit: $UPSTREAM_HEAD"
echo "HEAD: $(git -C "$PYTORCH_DIR" rev-parse HEAD)"
