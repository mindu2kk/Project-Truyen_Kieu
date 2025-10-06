# app/memory.py
# -*- coding: utf-8 -*-
"""
Bộ nhớ phiên làm việc rất nhẹ (in-memory), trích hồ sơ từ lời nói người dùng.
Hiện lưu: name. Có thể mở rộng thêm lớp/đề bài, sở thích, v.v.
"""

import re
from typing import Dict, Any

_MEM: Dict[str, Dict[str, Any]] = {}  # session_id -> profile


def get_profile(session_id: str | None) -> Dict[str, Any]:
    if not session_id:
        return {}
    return _MEM.get(session_id, {})


def set_profile(session_id: str | None, profile: Dict[str, Any]) -> None:
    if not session_id:
        return
    _MEM[session_id] = dict(profile or {})


def update_from_message(session_id: str | None, text: str) -> None:
    """Bắt các mẫu tự giới thiệu: 'tôi là ...', 'tên mình là ...'."""
    if not session_id or not text:
        return
    prof = get_profile(session_id).copy()

    # bắt tên: 'tôi|mình|tớ|em|anh|chị|tui ... (tên)'; ưu tiên 'tên là'
    patterns = [
        r"(?:tên\s*(?:của\s*tôi|mình)?\s*là|tôi\s*là|mình\s*là|tớ\s*là|em\s*là|anh\s*là|chị\s*là)\s+([A-Za-zÀ-ỹĐđ'’\-\s]{2,40})",
    ]
    name = None
    low = text.strip()

    for pat in patterns:
        m = re.search(pat, low, flags=re.I)
        if m:
            cand = m.group(1).strip()
            # rút gọn: tối đa 3 từ, bỏ kí tự thừa
            cand = re.sub(r"[^A-Za-zÀ-ỹĐđ'’\-\s]", "", cand).strip()
            parts = [p for p in cand.split() if p]
            if 1 <= len(parts) <= 3:
                name = " ".join(p.capitalize() for p in parts)
            else:
                # lấy 2–3 token cuối (tên thường ở cuối)
                if len(parts) > 3:
                    parts = parts[-2:]
                    name = " ".join(p.capitalize() for p in parts)
            break

    if name:
        prof["name"] = name

    set_profile(session_id, prof)


def profile_to_text(profile: Dict[str, Any]) -> str:
    """Chuẩn hoá hồ sơ sang chuỗi ngắn để nhúng vào prompt."""
    if not profile:
        return ""
    bits = []
    if profile.get("name"):
        bits.append(f"Tên người dùng: {profile['name']}.")
    return " ".join(bits)
