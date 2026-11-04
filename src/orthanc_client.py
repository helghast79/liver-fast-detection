"""Minimal Orthanc REST client with Basic Auth."""

from __future__ import annotations

import os
from typing import Any

import requests
from requests.auth import HTTPBasicAuth


class OrthancClient:
    def __init__(
        self,
        url: str | None = None,
        user: str | None = None,
        password: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = (url or os.getenv("ORTHANC_URL", "")).rstrip("/")
        if not self.base_url:
            raise ValueError("ORTHANC_URL is required")

        user = user if user is not None else os.getenv("ORTHANC_USER", "")
        password = password if password is not None else os.getenv("ORTHANC_PASSWORD", "")
        self.auth: HTTPBasicAuth | None = None
        if user or password:
            self.auth = HTTPBasicAuth(user, password)

        self.timeout = timeout
        self.session = requests.Session()
        self.session.auth = self.auth

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.base_url}{path}"

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        response = self.session.get(self._url(path), params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_bytes(self, path: str, params: dict[str, Any] | None = None) -> bytes:
        response = self.session.get(self._url(path), params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.content

    def post_json(self, path: str, payload: dict[str, Any]) -> Any:
        response = self.session.post(self._url(path), json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def find(
        self,
        level: str,
        query: dict[str, str],
        expand: bool = False,
        limit: int | None = None,
    ) -> list[Any]:
        payload: dict[str, Any] = {"Level": level, "Query": query}
        if expand:
            payload["Expand"] = True
        if limit is not None:
            payload["Limit"] = limit
        result = self.post_json("/tools/find", payload)
        if not isinstance(result, list):
            raise RuntimeError(f"Unexpected /tools/find response: {type(result)}")
        return result

    def get_patient(self, patient_id: str) -> dict[str, Any]:
        return self.get_json(f"/patients/{patient_id}")

    def get_study(self, study_id: str) -> dict[str, Any]:
        return self.get_json(f"/studies/{study_id}")

    def get_series(self, series_id: str) -> dict[str, Any]:
        return self.get_json(f"/series/{series_id}")

    def get_instance(self, instance_id: str) -> dict[str, Any]:
        return self.get_json(f"/instances/{instance_id}")

    def get_instance_tags(self, instance_id: str, simplify: bool = True) -> dict[str, Any]:
        if simplify:
            return self.get_json(f"/instances/{instance_id}/simplified-tags")
        return self.get_json(f"/instances/{instance_id}/tags")

    def download_instance_file(self, instance_id: str) -> bytes:
        return self.get_bytes(f"/instances/{instance_id}/file")

    def download_rendered(
        self,
        instance_id: str,
        width: int | None = None,
        height: int | None = None,
    ) -> bytes:
        params: dict[str, Any] = {}
        if width is not None:
            params["width"] = width
        if height is not None:
            params["height"] = height
        return self.get_bytes(f"/instances/{instance_id}/rendered", params=params or None)

    def find_instance_by_sop_uid(self, sop_instance_uid: str) -> str | None:
        matches = self.find("Instance", {"SOPInstanceUID": sop_instance_uid})
        if not matches:
            return None
        return matches[0]

    def find_series_by_uid(self, series_instance_uid: str) -> str | None:
        matches = self.find("Series", {"SeriesInstanceUID": series_instance_uid})
        if not matches:
            return None
        return matches[0]
