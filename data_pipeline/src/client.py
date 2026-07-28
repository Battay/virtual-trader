"""HTTP client for the PSX historical market endpoint."""

from datetime import date

import requests

from .config import PSX_HISTORICAL_URL, REQUEST_TIMEOUT_SECONDS


class PsxClientError(RuntimeError):
    """Raised when historical market data cannot be fetched."""


class PsxClient:
    """Fetch historical PSX market HTML using a reusable HTTP session."""

    def __init__(
        self,
        session: requests.Session | None = None,
        timeout: int = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self.session = session or requests.Session()
        self.timeout = timeout
        self.session.headers.update(
            {
                "User-Agent": "virtual-trader-fyp/0.1 (+university research)",
                "Accept": "text/html",
                "X-Requested-With": "XMLHttpRequest",
            }
        )

    def fetch_market_by_date(self, trading_date: date) -> str:
        """Return the HTML market rows published for ``trading_date``."""
        try:
            response = self.session.post(
                PSX_HISTORICAL_URL,
                data={"date": trading_date.isoformat()},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise PsxClientError(
                f"PSX request timed out after {self.timeout} seconds "
                f"for {trading_date.isoformat()}"
            ) from exc
        except requests.RequestException as exc:
            raise PsxClientError(
                f"PSX request failed for {trading_date.isoformat()}: {exc}"
            ) from exc

        if not response.text.strip():
            raise PsxClientError(
                f"PSX returned an empty response for {trading_date.isoformat()}"
            )
        return response.text
