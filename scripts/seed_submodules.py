#!/usr/bin/env python3
"""Seed submodules of an assembled source tree from a local reference tree.

Offline counterpart of `git submodule update --init --recursive`: compares
gitlinks of the reference tree against the assembled tree, copies matching
submodule working trees, strips the reference tree's .git metadata (linked
worktree pointers break when copied), and rewrites gitdir pointers to the
reference tree's local gitdirs so `git submodule` works offline afterwards.

Usage: seed_submodules.py <REFERENCE_REPO> <PYTORCH_DIR>
"""
import os
import shutil
import subprocess
import sys


def links(repo):
    out = subprocess.check_output(
        ["git", "-C", repo, "ls-tree", "-r", "HEAD"], text=True
    )
    return {
        l.split(None, 3)[3]: l.split(None, 3)[2]
        for l in out.splitlines()
        if l.split(None, 3)[1] == "commit"
    }


def purge(root):
    for dirpath, dirnames, filenames in os.walk(root):
        if ".git" in dirnames:
            shutil.rmtree(os.path.join(dirpath, ".git"), ignore_errors=True)
            dirnames.remove(".git")
        if ".git" in filenames:
            os.remove(os.path.join(dirpath, ".git"))


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    V, W = sys.argv[1], sys.argv[2]

    v_links, w_links = links(V), links(W)
    copied = skipped = 0
    for path, wsha in sorted(w_links.items()):
        if v_links.get(path) != wsha:
            skipped += 1
            continue
        src, dst = os.path.join(V, path), os.path.join(W, path)
        if os.path.isdir(dst) and os.listdir(dst):
            continue
        if not os.path.isdir(src) or not os.listdir(src):
            skipped += 1
            continue
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(src, dst, symlinks=True)
        copied += 1
    print(f"seeded {copied} submodules, skipped {skipped}")

    # Rewrite gitdir pointers: purge stale metadata, then mirror the reference
    # tree's pointer topology with absolute, existence-checked targets.
    pairs = []
    for path in sorted(w_links):
        root = os.path.join(V, path)
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if ".git" in dirnames:
                dirnames.remove(".git")
            if ".git" in filenames:
                f = os.path.join(dirpath, ".git")
                content = open(f).read().strip()
                if not content.startswith("gitdir:"):
                    continue
                target = content[len("gitdir:"):].strip()
                if not os.path.isabs(target):
                    target = os.path.normpath(os.path.join(dirpath, target))
                if os.path.isdir(target):
                    pairs.append((f, target))
    for path in sorted(w_links):
        root = os.path.join(W, path)
        if os.path.isdir(root):
            purge(root)
    for vf, target in pairs:
        wf = os.path.join(W, os.path.relpath(vf, V))
        os.makedirs(os.path.dirname(wf), exist_ok=True)
        with open(wf, "w") as fh:
            fh.write(f"gitdir: {target}\n")
    print(f"wrote {len(pairs)} gitdir pointers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
