from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PairedRow10:
    pre_ct: str
    pre_mask: str
    pre_aux_1: str
    pre_aux_2: str
    post_ct: str
    post_mask: str
    action_text: str
    pair_id: str
    survival_time_months: float
    event_indicator: int


def read_paired_tsv_10col(tsv_path: str | Path) -> list[PairedRow10]:
    p = Path(tsv_path)
    rows: list[PairedRow10] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 10:
            raise ValueError(f"{p}:{lineno} 期望 10 列 TSV，实际 {len(parts)} 列")
        rows.append(
            PairedRow10(
                pre_ct=parts[0],
                pre_mask=parts[1],
                pre_aux_1=parts[2],
                pre_aux_2=parts[3],
                post_ct=parts[4],
                post_mask=parts[5],
                action_text=parts[6],
                pair_id=parts[7],
                survival_time_months=float(parts[8]),
                event_indicator=int(float(parts[9])),
            )
        )
    return rows

