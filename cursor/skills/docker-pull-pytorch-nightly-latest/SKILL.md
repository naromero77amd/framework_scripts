---
name: docker-pull-pytorch-nightly-latest
description: Pull the ROCm PyTorch nightly Docker image only after verifying that rocm/pytorch-nightly:latest is not stale. Use when the user asks to pull, refresh, update, or verify the latest ROCm PyTorch nightly image.
---

# Docker Pull PyTorch Nightly Latest

## Goal

Pull this Docker image:

```bash
docker pull rocm/pytorch-nightly:latest
```

Before pulling, make sure the `latest` tag is not stale and really points at the latest available ROCm PyTorch nightly image.

## Freshness Rule

Do not trust the local Docker cache or assume the `latest` tag is current.

Before running `docker pull rocm/pytorch-nightly:latest`, inspect live Docker Hub tag metadata for `rocm/pytorch-nightly`:

1. Fetch metadata for `latest`.
2. Fetch the repository tag list.
3. Find the newest non-`latest` tag by `last_updated`.
4. Compare its image digest set with the `latest` digest set.
5. If a newer non-`latest` tag exists and has different digests than `latest`, stop and warn the user that `latest` appears stale. Do not pull `latest` unless the user explicitly approves.

If the Docker Hub metadata cannot be checked, stop and ask the user whether to proceed without the freshness guarantee.

## Verification Command

Use this Python script when shell access is available:

```bash
python3 - <<'PY'
import sys
import urllib.request
import json

base = "https://hub.docker.com/v2/repositories/rocm/pytorch-nightly/tags"

def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)

def digests(tag):
    return {
        image.get("digest")
        for image in tag.get("images", [])
        if image.get("digest")
    }

latest = fetch(f"{base}/latest")
tags = []
url = f"{base}?page_size=100"

while url:
    page = fetch(url)
    tags.extend(page.get("results", []))
    url = page.get("next")

candidates = [
    tag for tag in tags
    if tag.get("name") != "latest" and tag.get("last_updated")
]
newest = max(candidates, key=lambda tag: tag["last_updated"], default=None)

print(f"latest last_updated: {latest.get('last_updated')}")
print(f"latest digests: {sorted(digests(latest))}")

if newest:
    print(f"newest non-latest tag: {newest.get('name')}")
    print(f"newest non-latest last_updated: {newest.get('last_updated')}")
    print(f"newest non-latest digests: {sorted(digests(newest))}")

if newest and newest.get("last_updated", "") > latest.get("last_updated", "") and digests(newest) != digests(latest):
    print("ERROR: rocm/pytorch-nightly:latest appears stale.", file=sys.stderr)
    sys.exit(2)

print("OK: rocm/pytorch-nightly:latest appears current.")
PY
```

Only proceed if the script exits successfully.

## Pull Command

After freshness verification passes, execute:

```bash
docker pull rocm/pytorch-nightly:latest
```

Then report the pulled image digest if Docker provides one:

```bash
docker image inspect --format '{{json .RepoDigests}}' rocm/pytorch-nightly:latest
```
