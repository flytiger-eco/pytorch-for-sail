#!/usr/bin/env bash
# setup.sh - assemble the CUDA-compat PyTorch-for-SAIL source tree.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PATCH_DIR="$REPO_ROOT/patches"

# Work directory can be overridden via WORK env variable.
WORK="${WORK:-$REPO_ROOT/work}"
PYTORCH_DIR="$WORK/pytorch"

UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/t-head/pytorch-for-sail.git}"
UPSTREAM_BRANCH="${UPSTREAM_BRANCH:-v2.14.1}"
# Pinned base commit for the compat patches: the v2.14.1 branch head
# (8b41b061e474) as merged by PR #45.  The patches are generated against
# this exact commit, so builds check it out explicitly instead of trusting
# a possibly-moved branch head.
UPSTREAM_COMMIT="${UPSTREAM_COMMIT:-8b41b061e474d601f5333ff40547119fbe825c25}"
REFERENCE_REPO="${REFERENCE_REPO:-}"
# Common gitdir of the reference repo, when one is configured.  Used both to
# accelerate the clone and to fetch the pinned base commit offline.
REF_GIT_DIR=""
if [ -n "$REFERENCE_REPO" ] && [ -e "$REFERENCE_REPO/.git" ]; then
    REF_GIT_DIR=$(git -C "$REFERENCE_REPO" rev-parse --git-common-dir)
fi

mkdir -p "$WORK"

if [ -d "$PYTORCH_DIR/.git" ]; then
    echo "Reusing existing pytorch checkout at $PYTORCH_DIR"
    cd "$PYTORCH_DIR"
    git fetch origin "$UPSTREAM_BRANCH" \
        || echo "WARNING: could not fetch $UPSTREAM_BRANCH from origin; continuing with local objects"
else
    if [ -n "$REF_GIT_DIR" ]; then
        # Offline fast path: clone straight from the reference repo's object
        # store (hardlinked on the same filesystem, so no extra disk cost),
        # then re-point origin at the upstream URL.  This avoids the large
        # fetch from the upstream host entirely; the pinned-commit check
        # below then needs no network.
        echo "Cloning offline from reference repo $REFERENCE_REPO"
        git clone "$REF_GIT_DIR" "$PYTORCH_DIR"
        git -C "$PYTORCH_DIR" remote set-url origin "$UPSTREAM_URL"
    else
        echo "Cloning $UPSTREAM_URL branch $UPSTREAM_BRANCH into $PYTORCH_DIR"
        # The upstream host serves this large repository over a connection
        # that can intermittently drop ("Empty reply from server"); retry a
        # few times before giving up.
        CLONE_OK=0
        for attempt in 1 2 3; do
            if git clone -b "$UPSTREAM_BRANCH" --single-branch "$UPSTREAM_URL" "$PYTORCH_DIR"; then
                CLONE_OK=1
                break
            fi
            echo "clone attempt $attempt failed; retrying..."
            rm -rf "$PYTORCH_DIR"
            sleep 5
        done
        if [ "$CLONE_OK" != 1 ]; then
            echo "ERROR: clone failed after 3 attempts" >&2
            exit 1
        fi
    fi
    cd "$PYTORCH_DIR"
fi

# Make sure the pinned commit object is present before checking it out.
# The branch clone/fetch above only retrieves the branch head's history;
# when the branch has moved past the pin (or an initial fetch raced the
# server), fetch the commit explicitly: from the local reference repo when
# one is configured (works offline), otherwise from the remote.
if ! git cat-file -e "$UPSTREAM_COMMIT^{commit}" 2>/dev/null; then
    if [ -n "$REF_GIT_DIR" ]; then
        echo "Fetching pinned commit $UPSTREAM_COMMIT from reference repo"
        git fetch "$REF_GIT_DIR" "$UPSTREAM_COMMIT"
    else
        echo "Fetching pinned commit $UPSTREAM_COMMIT from origin"
        git fetch origin "$UPSTREAM_COMMIT"
    fi
fi

# Pin the base commit: check out the exact commit the patches were
# generated against, instead of trusting the (possibly moved) branch head.
if ! git checkout "$UPSTREAM_COMMIT"; then
    echo "ERROR: cannot check out pinned base commit $UPSTREAM_COMMIT" >&2
    echo "Make sure the commit is reachable from $UPSTREAM_URL branch $UPSTREAM_BRANCH." >&2
    exit 1
fi
echo "Pinned base commit: $(git rev-parse HEAD)"

# Ensure we are on a clean base.
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: pytorch worktree is not clean" >&2
    exit 1
fi

# Apply the compat patch series.
echo "Applying compat patches from $PATCH_DIR"
git am --abort 2>/dev/null || true
git am "$PATCH_DIR"/*.patch

# Seed submodules from a local reference tree to avoid full network clones.
# The seeder compares gitlinks against the reference tree, copies matching
# submodule working trees, strips the reference tree's .git metadata (linked
# worktree pointers break when copied), and rewrites gitdir pointers to the
# reference tree's local gitdirs so `git submodule` works offline afterwards.
if [ -n "$REFERENCE_REPO" ] && [ -e "$REFERENCE_REPO/.git" ]; then
    echo "Seeding submodules from reference repo $REFERENCE_REPO"
    python3.11 "$SCRIPT_DIR/seed_submodules.py" "$REFERENCE_REPO" "$PYTORCH_DIR"
else
    # No local reference tree: fall back to network submodule clones.
    echo "Syncing and updating submodules (this may take a while)"
    git submodule sync --recursive
    git submodule update --init --recursive
fi

# Inject the deployment Dockerfile into the source tree so the docker
# build workflow can use the source tree as its build context.  The file is
# added as an untracked, git-ignored asset: upstream ships its own docker/
# content (docker/sail) which must stay untouched, and ignoring the injected
# file keeps re-running setup.sh on the same tree clean.  The build context
# exclusions are handled by upstream's own .dockerignore (a tracked symlink
# to .gitignore), which already ignores build/ and dist/.
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
echo "HEAD: $(git -C "$PYTORCH_DIR" rev-parse HEAD)"
