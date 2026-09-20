from __future__ import annotations

import re

ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "e.g.", "i.e.", "vs.", "etc.", "u.s."}


def _protected_period(text: str, index: int) -> bool:
    if index > 0 and index + 1 < len(text) and text[index - 1].isdigit() and text[index + 1].isdigit():
        return True
    left = text[max(0, index - 12): index + 1].lower()
    if any(left.endswith(a) for a in ABBREVIATIONS):
        return True
    # Internal URL/email periods are followed by a non-space and therefore never
    # reach the boundary rule. A final period after a URL is real punctuation.
    return False


def _chunk(start: int, end: int, text: str, maximum: int) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    cursor = start
    while end - cursor > maximum:
        cut = text.rfind(" ", cursor, cursor + maximum + 1)
        if cut <= cursor:
            cut = cursor + maximum
        out.append((text[cursor:cut], cursor, cut))
        cursor = cut
        while cursor < end and text[cursor].isspace():
            cursor += 1
    if cursor < end:
        out.append((text[cursor:end], cursor, end))
    return out


def split_sentences(text: str, maximum: int = 500) -> list[tuple[str, int, int]]:
    if not isinstance(text, str):
        raise TypeError("turn content must be text")
    try:
        spans: list[tuple[int, int]] = []
        start = 0
        for i, char in enumerate(text):
            boundary = char in "!?" or (char == "." and not _protected_period(text, i))
            boundary = boundary and (i + 1 == len(text) or text[i + 1].isspace() or text[i + 1] in "\"'”’")
            if char == "\n" and i > start and (text[start:i].strip().endswith((':',)) or re.match(r"^\s*(?:[-*•]|\d+[.)])\s", text[start:i])):
                boundary = True
            if boundary:
                end = i + 1
                while start < end and text[start].isspace():
                    start += 1
                if start < end:
                    spans.append((start, end))
                start = end
        while start < len(text) and text[start].isspace():
            start += 1
        if start < len(text):
            spans.append((start, len(text)))
        result: list[tuple[str, int, int]] = []
        for begin, end in spans:
            while end > begin and text[end - 1].isspace():
                end -= 1
            result.extend(_chunk(begin, end, text, maximum))
        return result or ([(text, 0, len(text))] if text else [])
    except Exception:
        return [(text, 0, len(text))] if text else []
