import os
from typing import Union, TypeVar, Iterable, Optional
from pathlib import Path
from urllib.parse import urlparse

import torch
from timm.models.hub import download_cached_file, get_cache_dir
from huggingface_hub import try_to_load_from_cache
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE


Device = Union[str, torch.device]
Devices = Iterable[Device]
T = TypeVar('T')


def cached_path(url: str) -> str:
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    return os.path.join(get_cache_dir(), filename)


def cached_file(
    url: str, check_hash: bool = True, progress: bool = True
) -> str:
    path = cached_path(url)
    if not os.path.exists(path):
        print(f'Downloading model from {url!r} to {path!r}.')
        download_cached_file(url, check_hash, progress)
    return path


def hf_cached_root(
    repo_id: str,
    cache_dir: Union[str, Path, None] = None,
    revision: Optional[str] = None,
    repo_type: Optional[str] = None,
) -> str:
    if revision is None:
        revision = "main"
    if repo_type is None:
        repo_type = "model"
    if cache_dir is None:
        cache_dir = HUGGINGFACE_HUB_CACHE

    object_id = repo_id.replace("/", "--")
    repo_cache = os.path.join(cache_dir, f"{repo_type}s--{object_id}")
    if not os.path.isdir(repo_cache):
        # No cache for this model
        return repo_id

    refs_dir = os.path.join(repo_cache, "refs")
    snapshots_dir = os.path.join(repo_cache, "snapshots")
    # no_exist_dir = os.path.join(repo_cache, ".no_exist")

    # Resolve refs (for instance to convert main to the associated commit sha)
    if os.path.isdir(refs_dir):
        revision_file = os.path.join(refs_dir, revision)
        if os.path.isfile(revision_file):
            with open(revision_file) as f:
                revision = f.read()

    # Check if revision folder exists
    if not os.path.exists(snapshots_dir):
        return repo_id

    cached_shas = os.listdir(snapshots_dir)
    if revision not in cached_shas:
        # No cache for this revision and we won't try to return a random revision
        return repo_id

    # Check if dir exists in cache
    cached_root = os.path.join(snapshots_dir, revision)
    return cached_root if os.path.isdir(cached_root) else None
