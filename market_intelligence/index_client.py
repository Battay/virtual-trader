"""HTTP client for the official PSX end-of-day index time series."""

import logging
from typing import Any

import requests

from .index_config import (
    INDEX_ENDPOINT_TEMPLATE,
    INDEX_TIMEOUT_SECONDS,
    require_supported_index,
)

LOGGER = logging.getLogger(__name__)


class IndexClientError(RuntimeError):
    """Raised when an official index series cannot be acquired safely."""


class PsxIndexClient:
    def __init__(
        self,
        session: requests.Session | None = None,
        *,
        timeout: int = INDEX_TIMEOUT_SECONDS,
    ) -> None:
        self.session = session or requests.Session()
        self.timeout = timeout
        self.session.headers.update(
            {
                "User-Agent": "virtual-trader-fyp/0.1 (+university research)",
                "Accept": "application/json",
            }
        )

    def fetch_index_series(self, index_code: str) -> dict[str, Any]:
        definition = require_supported_index(index_code)
        url = INDEX_ENDPOINT_TEMPLATE.format(index_code=definition.code)
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.Timeout as exc:
            raise IndexClientError(
                f"PSX index request timed out after {self.timeout} seconds for "
                f"{definition.code}"
            ) from exc
        except requests.RequestException as exc:
            raise IndexClientError(
                f"PSX index request failed for {definition.code}: {exc}"
            ) from exc
        content_type = response.headers.get("Content-Type", "")
        if "application/json" not in content_type.lower():
            raise IndexClientError(
                f"PSX returned unexpected content type for {definition.code}: "
                f"{content_type or 'missing'}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise IndexClientError(
                f"PSX returned invalid JSON for {definition.code}"
            ) from exc
        if not isinstance(payload, dict):
            raise IndexClientError("PSX index response must be a JSON object")
        if payload.get("status") != 1:
            raise IndexClientError(
                f"PSX index response failed for {definition.code}: "
                f"{payload.get('message') or 'status was not 1'}"
            )
        data = payload.get("data")
        if not isinstance(data, list):
            raise IndexClientError("PSX index response data must be a list")
        if not data:
            raise IndexClientError(
                f"PSX returned no observations for supported index {definition.code}"
            )
        LOGGER.info("Fetched %s observations for %s", len(data), definition.code)
        return payload
