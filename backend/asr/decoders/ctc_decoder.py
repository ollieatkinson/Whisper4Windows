"""
CTC decoder scaffold.
"""

from __future__ import annotations

import numpy as np


def greedy_decode(logits: np.ndarray, token_table: list[str], blank_id: int = 0) -> str:
    """
    Placeholder CTC greedy decode implementation for future ONNX integration.
    """
    if logits.ndim < 2:
        return ""

    token_ids = np.argmax(logits, axis=-1).tolist()
    collapsed: list[int] = []
    prev = None
    for token_id in token_ids:
        if token_id == prev:
            continue
        prev = token_id
        if token_id == blank_id:
            continue
        collapsed.append(int(token_id))

    chars: list[str] = []
    for idx in collapsed:
        if 0 <= idx < len(token_table):
            chars.append(token_table[idx])
    return "".join(chars).strip()
