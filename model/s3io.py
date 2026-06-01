"""
model.s3io
==========
Thin boto3 helpers for reading/writing training artifacts from/to S3.
Used by DiagnosticsWriter, ProbeEvaluator, and the dashboards when
run_dir is an ``s3://`` URI.  Not imported on non-S3 (local) runs.
"""
from __future__ import annotations

import json
from typing import Any

_REGION = "us-west-2"


_FALLBACK_PROFILES = ("hack-scripps",)
_client_cache: "Any | None" = None


def _client():
    global _client_cache
    if _client_cache is not None:
        return _client_cache
    import boto3
    import os
    profile = os.environ.get("AWS_PROFILE") or os.environ.get("SPINHANCE_AWS_PROFILE")
    if profile:
        _client_cache = boto3.Session(profile_name=profile).client("s3", region_name=_REGION)
        return _client_cache
    # No explicit profile — try default credentials first (works on EC2/Garibaldi).
    # If they don't exist, fall back to known SSO profiles so the local dashboard
    # works without requiring AWS_PROFILE to be set in the shell.
    try:
        client = boto3.client("s3", region_name=_REGION)
        client.get_bucket_location(Bucket="spinhance-data")  # cheap auth probe
        _client_cache = client
        return _client_cache
    except Exception:
        pass
    for fallback in _FALLBACK_PROFILES:
        try:
            _client_cache = boto3.Session(profile_name=fallback).client("s3", region_name=_REGION)
            return _client_cache
        except Exception:
            pass
    _client_cache = boto3.client("s3", region_name=_REGION)
    return _client_cache


def _parse(uri: str) -> tuple[str, str]:
    assert uri.startswith("s3://"), f"expected s3:// URI, got {uri!r}"
    rest = uri[5:]
    bucket, _, key = rest.partition("/")
    return bucket, key


# ── Write ──────────────────────────────────────────────────────────────────────

def put_json(uri: str, payload: dict[str, Any]) -> None:
    bucket, key = _parse(uri)
    body = json.dumps(payload, indent=2, sort_keys=True).encode()
    _client().put_object(Bucket=bucket, Key=key, Body=body,
                         ContentType="application/json")


def put_bytes(uri: str, data: bytes,
              content_type: str = "application/octet-stream") -> None:
    bucket, key = _parse(uri)
    _client().put_object(Bucket=bucket, Key=key, Body=data,
                         ContentType=content_type)


# ── Read ───────────────────────────────────────────────────────────────────────

def get_bytes(uri: str) -> bytes | None:
    from botocore.exceptions import ClientError
    bucket, key = _parse(uri)
    try:
        resp = _client().get_object(Bucket=bucket, Key=key)
        return resp["Body"].read()
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def get_text(uri: str, default: str | None = None) -> str | None:
    data = get_bytes(uri)
    return data.decode() if data is not None else default


def get_json(uri: str, default: Any = None) -> Any:
    data = get_bytes(uri)
    if data is None:
        return default
    try:
        return json.loads(data)
    except Exception:
        return default


# ── Delete ─────────────────────────────────────────────────────────────────────

def delete(uri: str) -> None:
    from botocore.exceptions import ClientError
    bucket, key = _parse(uri)
    try:
        _client().delete_object(Bucket=bucket, Key=key)
    except ClientError:
        pass


def delete_prefix(uri_prefix: str) -> None:
    """Delete all objects whose key starts with the given prefix."""
    bucket, prefix = _parse(uri_prefix)
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    client = _client()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if objects:
            client.delete_objects(Bucket=bucket,
                                  Delete={"Objects": objects, "Quiet": True})


# ── List ───────────────────────────────────────────────────────────────────────

def list_prefixes(uri_prefix: str) -> list[str]:
    """Return immediate child directory names (last component only, no trailing /)."""
    bucket, prefix = _parse(uri_prefix)
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    client = _client()
    paginator = client.get_paginator("list_objects_v2")
    names: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            names.append(cp["Prefix"].rstrip("/").rsplit("/", 1)[-1])
    return names


def list_keys(uri_prefix: str) -> list[str]:
    """Return relative key names (no prefix) for all objects under the given prefix."""
    bucket, prefix = _parse(uri_prefix)
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    client = _client()
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            if rel:
                keys.append(rel)
    return keys
