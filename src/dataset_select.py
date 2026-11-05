"""Select HCC patients / SEG series / liver segment(s) per project rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .orthanc_client import OrthancClient

SEG_DESC_999_SPLEEN_LIVER = "Seg of Spleen, Liver"
SEG_DESC_999_LIVER = "Seg of Liver"


@dataclass
class SegChoice:
    patient_id: str
    orthanc_patient_id: str
    orthanc_study_id: str
    orthanc_series_id: str
    series_number: int
    series_description: str
    series_instance_uid: str
    study_instance_uid: str
    liver_segment_numbers: list[int]  # DICOM SegmentNumber(s) to merge (OR)
    selection_reason: str

    @property
    def liver_segment_number(self) -> int | None:
        """Primary segment number (first of the merge set), if known yet."""
        return self.liver_segment_numbers[0] if self.liver_segment_numbers else None


def _main_tags(resource: dict[str, Any]) -> dict[str, Any]:
    return resource.get("MainDicomTags") or {}


def _series_number(tags: dict[str, Any]) -> int | None:
    raw = tags.get("SeriesNumber")
    if raw is None or raw == "":
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        return None


def _is_seg_series(series: dict[str, Any]) -> bool:
    tags = _main_tags(series)
    modality = str(tags.get("Modality", "")).upper()
    return modality == "SEG"


def liver_segments_for_series(series_number: int, series_description: str) -> tuple[list[int], str]:
    """
    Return (segment_numbers_to_merge, reason).

    - Series 300: merge all segments except 'Abdominal aorta' (resolved from SEG tags)
    - Series 999 'Seg of Spleen, Liver': segment 2 (Liver)
    - Series 999 'Seg of Liver': segment 1 (Liver)
    """
    description = series_description.strip()
    if series_number == 300:
        # Empty preferred list => resolve_segment_numbers selects all non-aorta segments.
        return (
            [],
            "SEG SeriesNumber 300; liver = merge all segments except 'Abdominal aorta'",
        )
    if series_number == 999 and description == SEG_DESC_999_SPLEEN_LIVER:
        return (
            [2],
            "SEG SeriesNumber 999 'Seg of Spleen, Liver'; liver = second segment",
        )
    if series_number == 999 and description == SEG_DESC_999_LIVER:
        return (
            [1],
            "SEG SeriesNumber 999 'Seg of Liver'; liver = first segment",
        )
    raise ValueError(f"Unsupported SEG series {series_number!r} desc={description!r}")


def _list_studies_for_patient(client: OrthancClient, orthanc_patient_id: str) -> list[dict[str, Any]]:
    patient = client.get_patient(orthanc_patient_id)
    studies: list[dict[str, Any]] = []
    for study_id in patient.get("Studies", []):
        studies.append(client.get_study(study_id))
    return studies


def _list_series_for_study(client: OrthancClient, study: dict[str, Any]) -> list[dict[str, Any]]:
    series_list: list[dict[str, Any]] = []
    for series_id in study.get("Series", []):
        series_list.append(client.get_series(series_id))
    return series_list


def _find_seg_candidates(
    series_list: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return (seg_300, seg_999) if present in this study.

    For 999, prefer 'Seg of Spleen, Liver' over 'Seg of Liver' if both exist.
    """
    seg_300: dict[str, Any] | None = None
    seg_999_spleen_liver: dict[str, Any] | None = None
    seg_999_liver: dict[str, Any] | None = None
    for series in series_list:
        if not _is_seg_series(series):
            continue
        tags = _main_tags(series)
        number = _series_number(tags)
        description = str(tags.get("SeriesDescription", "")).strip()
        if number == 300:
            seg_300 = series
        elif number == 999 and description == SEG_DESC_999_SPLEEN_LIVER:
            seg_999_spleen_liver = series
        elif number == 999 and description == SEG_DESC_999_LIVER:
            seg_999_liver = series
    return seg_300, seg_999_spleen_liver or seg_999_liver


def _choice_from_series(
    *,
    patient_id: str,
    orthanc_patient_id: str,
    study: dict[str, Any],
    series: dict[str, Any],
    series_number: int,
) -> SegChoice:
    tags = _main_tags(series)
    description = str(tags.get("SeriesDescription", "")).strip()
    segment_numbers, reason = liver_segments_for_series(series_number, description)
    study_tags = _main_tags(study)
    return SegChoice(
        patient_id=patient_id,
        orthanc_patient_id=orthanc_patient_id,
        orthanc_study_id=study["ID"],
        orthanc_series_id=series["ID"],
        series_number=series_number,
        series_description=description,
        series_instance_uid=str(tags.get("SeriesInstanceUID", "")),
        study_instance_uid=str(study_tags.get("StudyInstanceUID", "")),
        liver_segment_numbers=segment_numbers,
        selection_reason=reason,
    )


def select_seg_for_patient_id(
    client: OrthancClient,
    patient_id: str,
    orthanc_patient_ids: list[str],
) -> SegChoice | None:
    """
    Among Orthanc patient resources sharing the same PatientID string,
    prefer a study with SEG series 300; else an accepted SEG 999.
    """
    best_300: SegChoice | None = None
    best_999: SegChoice | None = None

    for orthanc_patient_id in orthanc_patient_ids:
        for study in _list_studies_for_patient(client, orthanc_patient_id):
            series_list = _list_series_for_study(client, study)
            seg_300, seg_999 = _find_seg_candidates(series_list)

            if seg_300 is not None and best_300 is None:
                best_300 = _choice_from_series(
                    patient_id=patient_id,
                    orthanc_patient_id=orthanc_patient_id,
                    study=study,
                    series=seg_300,
                    series_number=300,
                )

            if seg_999 is not None and best_999 is None:
                best_999 = _choice_from_series(
                    patient_id=patient_id,
                    orthanc_patient_id=orthanc_patient_id,
                    study=study,
                    series=seg_999,
                    series_number=999,
                )

    return best_300 or best_999


def discover_hcc_patients(client: OrthancClient) -> dict[str, list[str]]:
    """
    Find Orthanc patients whose PatientID starts with 'hcc' (case-insensitive).
    Returns mapping PatientID -> list of Orthanc patient IDs.
    """
    grouped: dict[str, list[str]] = {}

    try:
        matches = client.find("Patient", {"PatientID": "hcc*"}, expand=True)
    except Exception:
        matches = []

    if matches and isinstance(matches[0], dict):
        resources = matches
    else:
        resources = []
        if matches and isinstance(matches[0], str):
            for pid in matches:
                resources.append(client.get_patient(pid))
        else:
            for pid in client.get_json("/patients"):
                resources.append(client.get_patient(pid))

    for resource in resources:
        if isinstance(resource, str):
            resource = client.get_patient(resource)
        tags = _main_tags(resource)
        patient_id = str(tags.get("PatientID", ""))
        if not patient_id.lower().startswith("hcc"):
            continue
        orthanc_id = resource.get("ID")
        if not orthanc_id:
            continue
        grouped.setdefault(patient_id, [])
        if orthanc_id not in grouped[patient_id]:
            grouped[patient_id].append(orthanc_id)

    return grouped


def select_all_patients(client: OrthancClient) -> list[SegChoice]:
    grouped = discover_hcc_patients(client)
    choices: list[SegChoice] = []
    for patient_id, orthanc_ids in sorted(grouped.items()):
        choice = select_seg_for_patient_id(client, patient_id, orthanc_ids)
        if choice is not None:
            choices.append(choice)
    return choices
