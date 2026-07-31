from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
import re

from .models import OCRItem


NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def assign_grid(
    items: list[OCRItem],
    screen_width: int,
    screen_height: int,
    columns: int = 6,
    rows: int = 10,
) -> list[OCRItem]:
    groups: dict[str, list[OCRItem]] = defaultdict(list)
    for item in items:
        cx, cy = item.center
        column = min(columns - 1, max(0, int(cx / screen_width * columns)))
        row = min(rows - 1, max(0, int(cy / screen_height * rows)))
        item.grid = f"{chr(ord('A') + column)}{row + 1}"
        item.relative = ""
        groups[item.grid].append(item)

    for group in groups.values():
        if len(group) <= 1:
            continue
        x_values = [item.center[0] for item in group]
        y_values = [item.center[1] for item in group]
        horizontal = max(x_values) - min(x_values) >= max(y_values) - min(y_values)
        ordered = sorted(group, key=lambda item: item.center[0 if horizontal else 1])
        if len(ordered) == 2:
            labels = ["왼쪽", "오른쪽"] if horizontal else ["위쪽", "아래쪽"]
        elif len(ordered) == 3:
            labels = (
                ["왼쪽", "가운데", "오른쪽"]
                if horizontal
                else ["위쪽", "가운데", "아래쪽"]
            )
        else:
            labels = [f"{index + 1}번째" for index in range(len(ordered))]
        for item, label in zip(ordered, labels):
            item.relative = label
    return items


def vibration_interval_ms(
    pointer: tuple[float, float],
    target: OCRItem,
    screen_size: tuple[int, int],
) -> tuple[int, bool, float]:
    px, py = pointer
    tx, ty = target.center
    width, height = screen_size
    distance = ((px - tx) ** 2 + (py - ty) ** 2) ** 0.5
    normalized = min(1.0, distance / max(1.0, (width**2 + height**2) ** 0.5))
    inside = (
        target.x <= px <= target.x + target.width
        and target.y <= py <= target.y + target.height
    )
    if inside:
        return 80, True, normalized
    interval = int(120 + 1480 * (normalized**0.7))
    return interval, False, normalized


def screen_position_label(
    item: OCRItem,
    screen_size: tuple[int, int],
) -> str:
    width, height = screen_size
    x, y = item.center
    horizontal = "왼쪽" if x < width / 3 else "오른쪽" if x > width * 2 / 3 else ""
    vertical = "위" if y < height / 3 else "아래" if y > height * 2 / 3 else ""
    if horizontal and vertical:
        return f"{horizontal} {vertical}"
    if horizontal:
        return f"{horizontal} 가운데"
    if vertical:
        return f"가운데 {vertical}"
    return "가운데"


def grid_numeric_coordinate(item: OCRItem) -> tuple[int, int]:
    """Convert a tactile grid label such as D4 to the numeric pair (4, 4)."""

    label = item.grid.strip().upper()
    if len(label) < 2 or not label[0].isalpha() or not label[1:].isdigit():
        raise ValueError(f"올바르지 않은 격자 좌표입니다: {item.grid!r}")
    return ord(label[0]) - ord("A") + 1, int(label[1:])


def target_direction_label(
    pointer: tuple[float, float],
    target: OCRItem,
) -> str:
    px, py = pointer
    horizontal = ""
    vertical = ""
    if px < target.x:
        horizontal = "오른쪽"
    elif px > target.x + target.width:
        horizontal = "왼쪽"
    if py < target.y:
        vertical = "아래"
    elif py > target.y + target.height:
        vertical = "위"
    if horizontal and vertical:
        return f"{horizontal} {vertical}"
    return horizontal or vertical or "현재 위치"


def normalized_item_text(item: OCRItem) -> str:
    return NON_WORD.sub("", item.text).casefold()


def find_query_match(
    query: str,
    candidates: list[OCRItem],
    minimum_score: float = 0.62,
) -> tuple[int, float] | None:
    normalized_query = NON_WORD.sub("", query).casefold()
    if not normalized_query or not candidates:
        return None
    scored: list[tuple[float, int]] = []
    for index, item in enumerate(candidates):
        candidate = normalized_item_text(item)
        if not candidate:
            continue
        if candidate == normalized_query:
            score = 1.0
        elif normalized_query in candidate or candidate in normalized_query:
            score = 0.82 + 0.18 * min(len(candidate), len(normalized_query)) / max(
                len(candidate), len(normalized_query)
            )
        else:
            score = SequenceMatcher(None, normalized_query, candidate).ratio()
        scored.append((score, index))
    if not scored:
        return None
    score, index = max(scored, key=lambda value: (value[0], -value[1]))
    return (index, score) if score >= minimum_score else None


def find_same_text_item_index(
    previous: OCRItem | None,
    candidates: list[OCRItem],
) -> int | None:
    if previous is None or not candidates:
        return None
    previous_text = normalized_item_text(previous)
    matches = [
        (index, item)
        for index, item in enumerate(candidates)
        if normalized_item_text(item) == previous_text
    ]
    if not matches:
        return None
    px, py = previous.center
    return min(
        matches,
        key=lambda pair: (
            (pair[1].center[0] - px) ** 2 + (pair[1].center[1] - py) ** 2
        ),
    )[0]


def find_matching_item_index(
    previous: OCRItem | None,
    candidates: list[OCRItem],
) -> int | None:
    if previous is None or not candidates:
        return None
    previous_text = normalized_item_text(previous)
    exact = [
        (index, item)
        for index, item in enumerate(candidates)
        if normalized_item_text(item) == previous_text
    ]
    pool = exact or list(enumerate(candidates))
    px, py = previous.center
    index, _ = min(
        pool,
        key=lambda pair: (
            (pair[1].center[0] - px) ** 2 + (pair[1].center[1] - py) ** 2,
            0 if pair[1].grid == previous.grid else 1,
        ),
    )
    return index
