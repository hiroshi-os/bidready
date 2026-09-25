"""Document store. Local filesystem is the default. S3 is optional behind the same interface."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from bidready.config import Settings


def safe_filename(name: str) -> str:
    base = Path(name or "document").name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return (cleaned or "document")[:180]


class DocumentStore(Protocol):
    def put(self, case_id: str, filename: str, data: bytes) -> str: ...

    def read(self, uri: str) -> bytes: ...

    def path_for(self, uri: str) -> Path: ...


class LocalDocumentStore:
    scheme = "file"

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, case_id: str, filename: str, data: bytes) -> str:
        dest = self.root / "cases" / case_id / safe_filename(filename)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            stem, suffix = dest.stem, dest.suffix
            dest = dest.with_name(f"{stem}-{case_id[:6]}{suffix}")
        dest.write_bytes(data)
        return f"file://{dest}"

    def path_for(self, uri: str) -> Path:
        if not uri.startswith("file://"):
            raise ValueError(f"not a local document uri: {uri}")
        return Path(uri.removeprefix("file://"))

    def read(self, uri: str) -> bytes:
        return self.path_for(uri).read_bytes()


class S3DocumentStore:
    """S3-compatible store. Imported only when DOCUMENT_STORE=s3 and boto3 is installed."""

    scheme = "s3"

    def __init__(self, settings: Settings) -> None:
        if not settings.s3_bucket or not settings.s3_endpoint_url:
            raise RuntimeError("S3 document store requires S3_BUCKET and S3_ENDPOINT_URL")
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "S3 document store requires boto3. Install it in the environment that sets DOCUMENT_STORE=s3."
            ) from exc
        self.bucket = settings.s3_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key or None,
            aws_secret_access_key=settings.s3_secret_key or None,
            region_name=settings.s3_region or "us-east-1",
        )

    def put(self, case_id: str, filename: str, data: bytes) -> str:
        key = f"cases/{case_id}/{safe_filename(filename)}"
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)
        return f"s3://{self.bucket}/{key}"

    def read(self, uri: str) -> bytes:
        bucket, key = _split_s3(uri)
        response = self.client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()

    def path_for(self, uri: str) -> Path:
        data = self.read(uri)
        bucket, key = _split_s3(uri)
        dest = Path("/tmp/bidready-s3") / bucket / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest


def _split_s3(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3 uri: {uri}")
    rest = uri.removeprefix("s3://")
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"incomplete s3 uri: {uri}")
    return bucket, key


def build_store(settings: Settings) -> DocumentStore:
    if settings.document_store == "local":
        return LocalDocumentStore(settings.data_dir)
    if settings.document_store == "s3":
        return S3DocumentStore(settings)
    raise RuntimeError(f"unknown DOCUMENT_STORE: {settings.document_store}")
