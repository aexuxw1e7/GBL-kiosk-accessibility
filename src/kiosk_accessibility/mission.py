from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


Point = tuple[float, float]
Rect = tuple[float, float, float, float]


class MissionPhase(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class MissionUpdate:
    recorded: bool
    completed: bool
    inside: bool
    wrong_click: bool = False


@dataclass(frozen=True)
class MissionResult:
    target_text: str
    elapsed_seconds: float
    wrong_clicks: int
    path_length_px: float
    direct_distance_px: float
    efficiency: float | None
    completion_source: str | None
    route_segments: tuple[tuple[Point, ...], ...]


@dataclass
class _RouteState:
    segments: list[list[Point]] = field(default_factory=list)
    total_distance: float = 0.0
    last_sample_at: float | None = None

    def current_segment(self) -> list[Point]:
        if not self.segments:
            self.segments.append([])
        return self.segments[-1]

    def break_segment(self) -> None:
        if self.segments and self.segments[-1]:
            self.segments.append([])
        self.last_sample_at = None


def clamp_point(point: Point, screen_size: tuple[int, int]) -> Point:
    width, height = screen_size
    return (
        min(max(float(point[0]), 0.0), max(0.0, float(width - 1))),
        min(max(float(point[1]), 0.0), max(0.0, float(height - 1))),
    )


def point_in_rect(point: Point, rect: Rect) -> bool:
    px, py = point
    x, y, width, height = rect
    return x <= px <= x + width and y <= py <= y + height


def distance_to_rect(point: Point, rect: Rect) -> float:
    px, py = point
    x, y, width, height = rect
    dx = max(x - px, 0.0, px - (x + width))
    dy = max(y - py, 0.0, py - (y + height))
    return math.hypot(dx, dy)


def path_length(points: tuple[Point, ...] | list[Point]) -> float:
    return sum(
        math.dist(previous, current)
        for previous, current in zip(points, points[1:])
    )


def path_efficiency(
    points: tuple[Point, ...] | list[Point],
    target_rect: Rect,
) -> float | None:
    if not points:
        return None
    direct = distance_to_rect(points[0], target_rect)
    travelled = path_length(points)
    if travelled <= 1e-9:
        return 1.0 if direct <= 1e-9 else 0.0
    return min(1.0, max(0.0, direct / travelled))


def same_target_identity(
    current_text: str,
    current_rect: Rect,
    candidate_text: str,
    candidate_rect: Rect,
    max_center_shift: float = 80.0,
    min_overlap_ratio: float = 0.25,
) -> bool:
    """Return whether an OCR candidate is the same physical target.

    Text alone is not a stable identity because one menu name can appear more than
    once on the screen. Routine OCR jitter is accepted only when the old and new
    text boxes still overlap, so a nearby duplicate cannot replace the target.
    """
    current_key = "".join(current_text.casefold().split())
    candidate_key = "".join(candidate_text.casefold().split())
    if not current_key or current_key != candidate_key:
        return False
    current_x, current_y, current_width, current_height = current_rect
    candidate_x, candidate_y, candidate_width, candidate_height = candidate_rect
    current_center = (
        current_x + current_width / 2.0,
        current_y + current_height / 2.0,
    )
    candidate_center = (
        candidate_x + candidate_width / 2.0,
        candidate_y + candidate_height / 2.0,
    )
    center_is_close = math.dist(current_center, candidate_center) <= max(
        0.0, float(max_center_shift)
    )
    intersection_width = max(
        0.0,
        min(current_x + current_width, candidate_x + candidate_width)
        - max(current_x, candidate_x),
    )
    intersection_height = max(
        0.0,
        min(current_y + current_height, candidate_y + candidate_height)
        - max(current_y, candidate_y),
    )
    current_area = max(0.0, current_width) * max(0.0, current_height)
    candidate_area = max(0.0, candidate_width) * max(0.0, candidate_height)
    smaller_area = min(current_area, candidate_area)
    if smaller_area <= 1e-9:
        return False
    overlap_ratio = intersection_width * intersection_height / smaller_area
    return center_is_close and overlap_ratio >= max(
        0.0, min(1.0, float(min_overlap_ratio))
    )


class MissionSession:
    def __init__(
        self,
        target_text: str,
        target_rect: Rect,
        screen_size: tuple[int, int],
        started_at: float,
        min_step_px: float = 5.0,
        min_sample_interval: float = 0.08,
    ) -> None:
        self.target_text = target_text
        self.target_rect = target_rect
        self.screen_size = screen_size
        self.started_at = float(started_at)
        self.min_step_px = max(0.0, float(min_step_px))
        self.min_sample_interval = max(0.0, float(min_sample_interval))
        self.phase = MissionPhase.RUNNING
        self.completed_at: float | None = None
        self.completion_source: str | None = None
        self.wrong_clicks = 0
        self._routes: dict[str, _RouteState] = {}

    @property
    def running(self) -> bool:
        return self.phase is MissionPhase.RUNNING

    @property
    def succeeded(self) -> bool:
        return self.phase is MissionPhase.SUCCEEDED

    def update_target(self, target_rect: Rect) -> None:
        if self.running:
            self.target_rect = target_rect

    def break_route(self, source: str | None = None) -> None:
        if source is None:
            for route in self._routes.values():
                route.break_segment()
            return
        route = self._routes.get(source)
        if route is not None:
            route.break_segment()

    def cancel(self) -> None:
        if self.running:
            self.phase = MissionPhase.CANCELLED

    def _record_point(
        self,
        point: Point,
        now: float,
        source: str,
        force: bool,
        inside: bool,
    ) -> bool:
        route = self._routes.setdefault(source, _RouteState())
        segment = route.current_segment()
        if not segment:
            segment.append(point)
            route.last_sample_at = now
            return True

        distance = math.dist(segment[-1], point)
        enough_time = (
            route.last_sample_at is None
            or now - route.last_sample_at >= self.min_sample_interval
        )
        if not force and not inside and (
            distance < self.min_step_px or not enough_time
        ):
            return False
        route.total_distance += distance
        segment.append(point)
        route.last_sample_at = now
        return True

    def observe(
        self,
        point: Point,
        now: float,
        source: str,
        complete_on_inside: bool = True,
        force_sample: bool = False,
    ) -> MissionUpdate:
        bounded = clamp_point(point, self.screen_size)
        inside = point_in_rect(bounded, self.target_rect)
        if not self.running:
            return MissionUpdate(False, False, inside)
        recorded = self._record_point(
            bounded,
            float(now),
            source,
            force_sample,
            inside,
        )
        completed = False
        if complete_on_inside and inside:
            self.phase = MissionPhase.SUCCEEDED
            self.completed_at = max(float(now), self.started_at)
            self.completion_source = source
            completed = True
        return MissionUpdate(recorded, completed, inside)

    def register_click(
        self,
        point: Point,
        now: float,
        source: str = "cursor",
    ) -> MissionUpdate:
        if not self.running:
            inside = point_in_rect(
                clamp_point(point, self.screen_size), self.target_rect
            )
            return MissionUpdate(False, False, inside)
        update = self.observe(
            point,
            now,
            source,
            complete_on_inside=True,
            force_sample=True,
        )
        wrong_click = not update.inside
        if wrong_click:
            self.wrong_clicks += 1
        return MissionUpdate(
            update.recorded,
            update.completed,
            update.inside,
            wrong_click,
        )

    def route_segments(self) -> dict[str, tuple[tuple[Point, ...], ...]]:
        return {
            source: tuple(tuple(segment) for segment in route.segments if segment)
            for source, route in self._routes.items()
        }

    def snapshot(self, now: float) -> MissionResult:
        finished_at = self.completed_at if self.completed_at is not None else float(now)
        elapsed = max(0.0, finished_at - self.started_at)
        source = self.completion_source
        if source is None and self._routes:
            source = max(
                self._routes,
                key=lambda key: sum(len(part) for part in self._routes[key].segments),
            )
        route = self._routes.get(source) if source is not None else None
        segments = (
            tuple(tuple(segment) for segment in route.segments if segment)
            if route is not None
            else ()
        )
        flattened = tuple(point for segment in segments for point in segment)
        travelled = route.total_distance if route is not None else 0.0
        direct = (
            distance_to_rect(flattened[0], self.target_rect)
            if flattened
            else 0.0
        )
        if not flattened:
            efficiency = None
        elif travelled <= 1e-9:
            efficiency = 1.0 if direct <= 1e-9 else 0.0
        else:
            efficiency = min(1.0, max(0.0, direct / travelled))
        return MissionResult(
            target_text=self.target_text,
            elapsed_seconds=elapsed,
            wrong_clicks=self.wrong_clicks,
            path_length_px=travelled,
            direct_distance_px=direct,
            efficiency=efficiency,
            completion_source=self.completion_source,
            route_segments=segments,
        )
