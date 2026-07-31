from __future__ import annotations

import time
import tkinter as tk
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from statistics import median
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from .analyzer import KioskAnalyzer, annotate_warped
from .camera import (
    CameraOpenResult,
    enumerate_camera_devices,
    frame_is_blank,
    open_camera,
)
from .demo import create_mock_kiosk
from .grid import (
    find_query_match,
    find_same_text_item_index,
    grid_numeric_coordinate,
    screen_position_label,
    target_direction_label,
    vibration_interval_ms,
)
from .ocr import StaticOCR, TesseractOCR
from .speech import Speaker
from .tracking import (
    ApproachRateTracker,
    draw_pointer_to_target,
    draw_source_tracking,
    draw_warped_pointer,
    frame_point_to_screen,
    track_camera_center,
)
from .vision import (
    MIN_OCR_FRAME_QUALITY,
    ScreenDetectionTracker,
    StabilityTracker,
    alignment_message,
    detect_screen,
    draw_detection,
    full_frame_detection,
    ocr_frame_quality,
    screen_is_fully_visible,
    screen_change_score,
    select_best_ocr_candidate,
    warp_screen,
)


class PrototypeApp:
    def __init__(self, root: tk.Tk, project_root: Path) -> None:
        self.root = root
        self.project_root = project_root
        self.ocr = TesseractOCR(project_root)
        self.analyzer = KioskAnalyzer(self.ocr)
        self.speaker = Speaker(enabled=True)
        self.stability = StabilityTracker(
            history=6,
            movement_threshold=0.018,
        )
        self.screen_tracker = ScreenDetectionTracker(max_recoveries=4)
        self.approach = ApproachRateTracker()
        self.camera_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gbl-camera"
        )
        self.analysis_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gbl-analysis"
        )
        self.capture = None
        self.camera_open_future: Future | None = None
        self.camera_generation = 0
        self.camera_read_failures = 0
        self.camera_blank_frames = 0
        self.camera_blank_reconnects = 0
        self.camera_description = ""
        self.camera_devices = enumerate_camera_devices(maximum_indices=8)
        self.current_camera_is_camo = False
        self.input_generation = 0
        self.analysis_future: Future | None = None
        self.pending_analysis = None
        self.analysis_in_progress = False
        self.closed = False
        self.current_frame = None
        self.current_detection = None
        self.current_frame_id = 0
        self.analysis_request_id = 0
        self.result = None
        self.selected_item = None
        self.pending_candidate_index = None
        self.manual_pointer = None
        self.location_voice_message = ""
        self.demo_items = None
        self.last_warp = None
        self.last_analysis_at = 0.0
        self.last_successful_analysis_at = 0.0
        self.frame_candidates = deque(maxlen=12)
        self.last_candidate_at = 0.0
        self.screen_miss_count = 0
        self.candidate_change_votes = 0
        self.screen_epoch = 0
        self.use_full_frame_tracking = False
        self.live_source_annotated = None
        self.live_corrected_annotated = None
        self.tracking_was_inside = False
        self.haptic_active = False
        self.haptic_inside = False
        self.haptic_interval = 1000
        self.haptic_detail = "목표를 선택하면 진동이 시작됩니다."
        self.last_spoken = ""
        self.last_spoken_at = 0.0
        self.photo = None
        self.display_geometry = None

        self.root.title("촉각 격자 키오스크 접근성 프로토타입")
        self.root.geometry("1320x900")
        self.root.minsize(1100, 760)
        self._build()
        self.root.after(80, self._haptic_tick)
        self.load_demo()

    def _build(self) -> None:
        style = ttk.Style()
        style.configure("Title.TLabel", font=("맑은 고딕", 18, "bold"))
        style.configure("Status.TLabel", font=("맑은 고딕", 12, "bold"))
        style.configure("TButton", font=("맑은 고딕", 10), padding=7)

        header = ttk.Frame(self.root, padding=(18, 14, 18, 10))
        header.pack(fill="x")
        ttk.Label(
            header,
            text="실시간 화면 인식 · OCR · 6×10 촉각 격자",
            style="Title.TLabel",
        ).pack(side="left")
        self.engine_label = ttk.Label(header, text=self.ocr.name)
        self.engine_label.pack(side="right")

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        viewer = ttk.Frame(body)
        controls = ttk.Frame(body, padding=(16, 4))
        body.add(viewer, weight=3)
        body.add(controls, weight=1)

        self.canvas = tk.Canvas(
            viewer, background="#17202A", highlightthickness=0, cursor="crosshair"
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda event: self._render_current())
        self.canvas.bind("<Motion>", self._pointer_motion)
        self.canvas.bind("<Button-1>", self._canvas_click)
        self.canvas.bind("<Leave>", lambda event: self.haptic_var.set("포인터를 보정 화면 위로 이동하세요."))

        source_bar = ttk.Frame(viewer, padding=(0, 8, 0, 0))
        source_bar.pack(fill="x")
        self.view_mode = tk.StringVar(value="corrected")
        ttk.Radiobutton(
            source_bar,
            text="원본·검출",
            variable=self.view_mode,
            value="source",
            command=self._render_current,
        ).pack(side="left")
        ttk.Radiobutton(
            source_bar,
            text="원근 보정·격자",
            variable=self.view_mode,
            value="corrected",
            command=self._render_current,
        ).pack(side="left", padx=(12, 0))
        self.haptic_var = tk.StringVar(value="메뉴를 선택하면 진동 주기를 시뮬레이션합니다.")
        ttk.Label(source_bar, textvariable=self.haptic_var).pack(side="right")

        ttk.Label(controls, text="1. 화면 입력", style="Status.TLabel").pack(
            anchor="w", pady=(0, 6)
        )
        input_buttons = ttk.Frame(controls)
        input_buttons.pack(fill="x")
        ttk.Button(input_buttons, text="내장 데모", command=self.load_demo).pack(
            side="left", expand=True, fill="x"
        )
        ttk.Button(input_buttons, text="사진 열기", command=self.open_image).pack(
            side="left", expand=True, fill="x", padx=5
        )
        self.camera_button = ttk.Button(
            input_buttons, text="웹캠 시작", command=self.toggle_camera
        )
        self.camera_button.pack(side="left", expand=True, fill="x")

        camera_options = ttk.Frame(controls)
        camera_options.pack(fill="x", pady=(7, 0))
        ttk.Label(camera_options, text="카메라:").pack(side="left")
        self.auto_camera_label = "자동 탐색 (Camo/iPhone 우선)"
        self.camera_source_indices: dict[str, int | None] = {
            self.auto_camera_label: None
        }
        listed_indices = set()
        for device in self.camera_devices:
            label = f"{device.display_name} (카메라 {device.index})"
            self.camera_source_indices[label] = device.index
            listed_indices.add(device.index)
        for index in range(4):
            if index not in listed_indices:
                self.camera_source_indices[f"카메라 {index}"] = index
        self.camera_source_var = tk.StringVar(value=self.auto_camera_label)
        self.camera_selector = ttk.Combobox(
            camera_options,
            textvariable=self.camera_source_var,
            values=list(self.camera_source_indices),
            state="readonly",
            width=25,
        )
        self.camera_selector.pack(side="left", padx=(6, 0), fill="x", expand=True)

        analysis_buttons = ttk.Frame(controls)
        analysis_buttons.pack(fill="x", pady=(8, 10))
        self.analyze_button = ttk.Button(
            analysis_buttons,
            text="화면 인식·다시 OCR (F5)",
            command=self.analyze_current,
        )
        self.analyze_button.pack(side="left", fill="x", expand=True)
        self.full_frame_button = ttk.Button(
            analysis_buttons,
            text="전체 프레임 OCR",
            command=lambda: self.analyze_current(force_full_frame=True),
        )
        self.full_frame_button.pack(side="left", fill="x", expand=True, padx=(5, 0))

        ttk.Label(
            controls, text="2. 카메라·화면 연결", style="Status.TLabel"
        ).pack(anchor="w")
        camo_available = any(device.is_camo for device in self.camera_devices)
        initial_connection = (
            "● Camo/iPhone 감지됨 · 연결 대기"
            if camo_available
            else "● 카메라 연결 대기"
        )
        self.connection_status_var = tk.StringVar(value=initial_connection)
        self.connection_status_label = ttk.Label(
            controls,
            textvariable=self.connection_status_var,
            wraplength=330,
            foreground="#6B7280",
        )
        self.connection_status_label.pack(
            anchor="w", fill="x", pady=(3, 2)
        )
        self.camera_status_var = tk.StringVar(value="카메라 미사용")
        ttk.Label(
            controls,
            textvariable=self.camera_status_var,
            wraplength=330,
            foreground="#31566F",
        ).pack(anchor="w", fill="x", pady=(0, 7))

        self.status_var = tk.StringVar(value="OCR 분석 대기")
        ttk.Label(
            controls,
            textvariable=self.status_var,
            wraplength=330,
            foreground="#0A6F57",
        ).pack(anchor="w", fill="x", pady=(0, 9))

        ttk.Label(controls, text="3. 찾을 메뉴 입력", style="Status.TLabel").pack(
            anchor="w"
        )
        target_input = ttk.Frame(controls)
        target_input.pack(fill="x", pady=(5, 5))
        self.target_query_var = tk.StringVar()
        self.target_entry = ttk.Entry(
            target_input,
            textvariable=self.target_query_var,
            font=("맑은 고딕", 11),
        )
        self.target_entry.pack(side="left", fill="x", expand=True)
        self.target_entry.bind("<Return>", self.search_target)
        ttk.Button(target_input, text="찾기·추적", command=self.search_target).pack(
            side="left", padx=(5, 0)
        )

        self.target_status_var = tk.StringVar(
            value="메뉴명을 입력하거나 아래 목록에서 목표를 선택하세요."
        )
        ttk.Label(
            controls,
            textvariable=self.target_status_var,
            wraplength=330,
            font=("맑은 고딕", 10, "bold"),
            foreground="#173B57",
        ).pack(anchor="w", fill="x", pady=(0, 5))

        self.haptic_panel = tk.Label(
            controls,
            text="진동 대기",
            font=("맑은 고딕", 11, "bold"),
            foreground="#34495E",
            background="#E9EEF2",
            padx=8,
            pady=7,
        )
        self.haptic_panel.pack(fill="x", pady=(0, 9))

        ttk.Label(
            controls,
            text="4. 인식된 메뉴 — 클릭 또는 방향키·Enter",
            style="Status.TLabel",
        ).pack(anchor="w")
        self.item_list = tk.Listbox(
            controls,
            font=("맑은 고딕", 11),
            height=9,
            activestyle="dotbox",
            exportselection=False,
        )
        self.item_list.pack(fill="both", expand=True, pady=(6, 8))
        self.item_list.bind("<<ListboxSelect>>", self.select_item)
        self.item_list.bind("<Return>", self.select_item)

        self.guide_var = tk.StringVar(value="분석 후 메뉴를 선택하세요.")
        guide = ttk.Label(
            controls,
            textvariable=self.guide_var,
            wraplength=330,
            font=("맑은 고딕", 11, "bold"),
            foreground="#173B57",
        )
        guide.pack(anchor="w", fill="x", pady=(0, 10))

        self.tts_var = tk.BooleanVar(value=True)
        speech_controls = ttk.Frame(controls)
        speech_controls.pack(fill="x")
        ttk.Checkbutton(
            speech_controls,
            text="자연 음성 사용 (R을 누를 때만 재생)",
            variable=self.tts_var,
            command=self._toggle_speech,
        ).pack(side="left", anchor="w")
        ttk.Button(
            speech_controls,
            text="안내 듣기 (R)",
            command=self.repeat_current_guidance,
        ).pack(side="right")

        note = (
            "녹색: 높은 신뢰도 · 주황: 확인 필요\n"
            "노란 상자: 카메라가 찾는 목표\n"
            "십자선: 카메라 중심점(가상 손가락)\n"
            "화면 클릭: 클릭 위치 기준 안내 갱신 · R로 듣기\n"
            "F2 카메라 · F3 사진 · F4 데모 · R 안내 듣기 · Esc 중지"
        )
        ttk.Label(controls, text=note, foreground="#566573").pack(
            anchor="w", pady=(10, 0)
        )

        self.root.bind("<F2>", lambda event: self.toggle_camera())
        self.root.bind("<F3>", lambda event: self.open_image())
        self.root.bind("<F4>", lambda event: self.load_demo())
        self.root.bind("<F5>", lambda event: self.analyze_current())
        self.root.bind("<Escape>", lambda event: self.stop_camera(announce=True))
        self.root.bind_all("<KeyPress>", self._global_keypress, add="+")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _set_connection_status(self, message: str, state: str) -> None:
        colors = {
            "idle": "#6B7280",
            "connecting": "#B26A00",
            "connected": "#0A7A4B",
            "warning": "#B45309",
            "error": "#B42318",
        }
        shown = message if message.startswith("●") else f"● {message}"
        self.connection_status_var.set(shown)
        self.connection_status_label.configure(
            foreground=colors.get(state, colors["idle"])
        )

    def _idle_connection_message(self) -> str:
        if any(device.is_camo for device in self.camera_devices):
            return "Camo/iPhone 감지됨 · 연결 대기"
        return "카메라 연결 대기"

    def _camera_label_for_index(self, index: int) -> str:
        for label, mapped_index in self.camera_source_indices.items():
            if mapped_index == index:
                return label
        return f"카메라 {index}"

    def _selected_camera_is_camo(self, preferred_index: int | None) -> bool:
        if preferred_index is None:
            return any(device.is_camo for device in self.camera_devices)
        return any(
            device.index == preferred_index and device.is_camo
            for device in self.camera_devices
        )

    def _toggle_speech(self) -> None:
        self.speaker.set_enabled(self.tts_var.get())

    def _global_keypress(self, event) -> str | None:
        if self.root.focus_get() is self.target_entry:
            return None
        is_r_key = (
            str(getattr(event, "keysym", "")).casefold() == "r"
            or getattr(event, "keycode", None) == 82
        )
        if not is_r_key:
            return None
        self.repeat_current_guidance(event)
        return "break"

    def _clear_frame_candidates(self) -> None:
        self.frame_candidates.clear()
        self.last_candidate_at = 0.0
        self.candidate_change_votes = 0

    def _record_frame_candidate(
        self,
        frame,
        detection,
        warped,
        now: float,
        movement: float,
        stable: bool,
        source: str,
        tracking_confidence: float,
        force_full_frame: bool,
    ) -> tuple[str, float]:
        quality = ocr_frame_quality(warped)
        fully_visible = force_full_frame or screen_is_fully_visible(
            detection,
            frame.shape,
        )
        if not fully_visible:
            return "clipped", quality
        if quality < MIN_OCR_FRAME_QUALITY:
            return "blurry", quality
        if not stable and movement > 0.038:
            return "shaking", quality
        if source == "tracked" and tracking_confidence < 0.58:
            return "tracking_low", quality
        if now - self.last_candidate_at < 0.12:
            return "waiting", quality

        if self.frame_candidates:
            change = screen_change_score(
                self.frame_candidates[-1]["warp"],
                warped,
            )
            if change > 0.085:
                self.candidate_change_votes += 1
                if self.candidate_change_votes < 2:
                    return "change_pending", quality
                self._clear_frame_candidates()
                self.screen_epoch += 1
            elif change < 0.045:
                self.candidate_change_votes = 0
            else:
                self.candidate_change_votes = max(
                    0,
                    self.candidate_change_votes - 1,
                )

        self.frame_candidates.append(
            {
                "quality": quality,
                "frame": frame.copy(),
                "detection": detection,
                "warp": warped.copy(),
                "timestamp": now,
                "movement": movement,
                "source": source,
                "tracking_confidence": tracking_confidence,
            }
        )
        self.last_candidate_at = now
        return "accepted", quality

    def _camera_candidates_ready(
        self,
        now: float,
        stable: bool,
        movement: float | None,
    ) -> bool:
        recent_count = sum(
            now - candidate["timestamp"] <= 2.0
            for candidate in self.frame_candidates
        )
        if recent_count < 3:
            return False
        if stable:
            return True
        recent_movements = [
            candidate["movement"]
            for candidate in self.frame_candidates
            if now - candidate["timestamp"] <= 2.0
            and candidate["movement"] < 1.0
        ]
        return (
            recent_count >= 5
            and recent_movements
            and median(recent_movements) <= 0.028
        )

    def _result_needs_retry(self) -> bool:
        if self.result is None or not self.result.items:
            return True
        reliable = [
            item for item in self.result.items if item.confidence >= 65
        ]
        return not reliable

    def _maybe_start_camera_analysis(
        self,
        now: float,
        stable: bool,
        movement: float | None,
        force_full_frame: bool,
    ) -> None:
        if (
            not self._camera_candidates_ready(now, stable, movement)
            or now - self.last_analysis_at <= 2.5
        ):
            return
        best_candidate = select_best_ocr_candidate(
            self.frame_candidates,
            now,
            max_age=2.0,
        )
        if best_candidate is None:
            return
        changed = screen_change_score(
            self.last_warp,
            best_candidate["warp"],
        )
        last_attempt = max(
            self.last_analysis_at,
            self.last_successful_analysis_at,
        )
        retry_due = (
            self._result_needs_retry()
            and now - last_attempt > 5.5
        )
        if self.last_warp is None or changed > 0.055 or retry_due:
            self.analyze_current(
                auto=True,
                force_full_frame=force_full_frame,
                detection=best_candidate["detection"],
                comparison_warp=best_candidate["warp"],
            )

    def load_demo(self) -> None:
        self.stop_camera()
        self._clear_frame_candidates()
        self.input_generation += 1
        self.pending_analysis = None
        self.current_frame, self.demo_items = create_mock_kiosk()
        self.current_detection = detect_screen(self.current_frame)
        self.current_frame_id += 1
        self.result = None
        self.selected_item = None
        self.pending_candidate_index = None
        self.use_full_frame_tracking = False
        self.live_source_annotated = None
        self.live_corrected_annotated = None
        self.item_list.delete(0, tk.END)
        self.target_query_var.set("")
        self._reset_tracking_feedback()
        self._set_connection_status(
            self._idle_connection_message(), "idle"
        )
        self.camera_status_var.set("내장 데모 사용 · 카메라 미사용")
        self.status_var.set("내장 모의 키오스크를 분석합니다.")
        self.analyze_current()

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="키오스크 화면 사진 선택",
            filetypes=[
                ("이미지", "*.png *.jpg *.jpeg *.bmp"),
                ("모든 파일", "*.*"),
            ],
        )
        if not path:
            return
        self.stop_camera()
        self._clear_frame_candidates()
        raw = Path(path).read_bytes()
        image = cv2.imdecode(
            __import__("numpy").frombuffer(raw, dtype="uint8"), cv2.IMREAD_COLOR
        )
        if image is None:
            messagebox.showerror("오류", "이미지를 열 수 없습니다.")
            return
        self.input_generation += 1
        self._clear_frame_candidates()
        self.pending_analysis = None
        self.current_frame = image
        self.current_detection = detect_screen(image)
        self.current_frame_id += 1
        self.demo_items = None
        self.result = None
        self.selected_item = None
        self.pending_candidate_index = None
        self.pending_candidate_index = None
        self.use_full_frame_tracking = False
        self.live_source_annotated = None
        self.live_corrected_annotated = None
        self._reset_tracking_feedback()
        self.item_list.delete(0, tk.END)
        self._set_connection_status(
            self._idle_connection_message(), "idle"
        )
        self.camera_status_var.set("사진 입력 · 카메라 미사용")
        self.status_var.set("사진을 불러왔습니다. 현재 화면 분석을 누르세요.")
        self.view_mode.set("source")
        self._render_current()

    def toggle_camera(self) -> None:
        if self.capture is not None or self.camera_open_future is not None:
            self.stop_camera(announce=True)
            return
        preferred = self._preferred_camera_index()
        automatic_selection = preferred is None
        self.input_generation += 1
        self.pending_analysis = None
        self.result = None
        self.selected_item = None
        self.pending_candidate_index = None
        self.current_detection = None
        self.live_source_annotated = None
        self.live_corrected_annotated = None
        self.use_full_frame_tracking = False
        self.item_list.delete(0, tk.END)
        self._reset_tracking_feedback()
        self.screen_tracker.reset()
        self.stability.reset()
        self.screen_miss_count = 0
        self._clear_frame_candidates()
        self.camera_generation += 1
        generation = self.camera_generation
        self.camera_button.configure(text="탐색 취소")
        self.camera_selector.configure(state="disabled")
        camo_available = any(device.is_camo for device in self.camera_devices)
        if automatic_selection and camo_available:
            self._set_connection_status(
                "Camo/iPhone 우선 연결 중...", "connecting"
            )
            self.camera_status_var.set(
                "Camo 장치를 먼저 확인한 뒤 다른 RGB 카메라를 탐색합니다."
            )
        else:
            self._set_connection_status("카메라 연결 중...", "connecting")
            self.camera_status_var.set("선택한 카메라의 영상을 확인하는 중...")
        self.status_var.set("카메라 연결 후 화면을 자동 분석합니다.")
        self._speak_throttled("사용 가능한 카메라를 찾는 중입니다.", force=True)
        open_arguments = {
            "preferred_index": preferred,
            "maximum_indices": 8,
            "warmup_reads": 6,
            "known_devices": self.camera_devices,
        }
        if self._selected_camera_is_camo(preferred):
            # Camo's DirectShow stream is black when this installation opens
            # it in a ThreadPool worker and later reads it on Tk's UI thread.
            # Opening and reading on the same Tk thread preserves its native
            # virtual-camera format.
            future = Future()
            self.camera_open_future = future
            self.root.update_idletasks()
            try:
                open_result = open_camera(**open_arguments)
            except Exception as error:
                future.set_exception(error)
            else:
                future.set_result(open_result)
        else:
            future = self.camera_executor.submit(
                open_camera,
                **open_arguments,
            )
            self.camera_open_future = future
        self.root.after(
            80,
            lambda: self._poll_camera_open(
                future, generation, automatic_selection
            ),
        )

    def _preferred_camera_index(self) -> int | None:
        value = self.camera_source_var.get()
        return self.camera_source_indices.get(value)

    def _poll_camera_open(
        self, future: Future, generation: int, automatic_selection: bool
    ) -> None:
        if self.closed:
            return
        if not future.done():
            self.root.after(
                80,
                lambda: self._poll_camera_open(
                    future, generation, automatic_selection
                ),
            )
            return
        if self.camera_open_future is future:
            self.camera_open_future = None
        try:
            open_result: CameraOpenResult = future.result()
        except Exception as error:
            open_result = CameraOpenResult(session=None, attempts=[])
            error_message = f"카메라 초기화 중 오류가 발생했습니다: {error}"
        else:
            error_message = open_result.error_message
        if generation != self.camera_generation:
            if open_result.session is not None:
                open_result.session.capture.release()
            return
        self.camera_selector.configure(state="readonly")
        if open_result.session is None:
            self.camera_button.configure(text="웹캠 시작")
            self._set_connection_status("카메라 연결 실패", "error")
            self.camera_status_var.set(error_message)
            self._speak_throttled("카메라를 열지 못했습니다. 다른 앱과 개인정보 설정을 확인하세요.", force=True)
            messagebox.showerror("웹캠 오류", error_message)
            return

        session = open_result.session
        self.capture = session.capture
        self.camera_description = session.description
        self.current_camera_is_camo = session.is_camo
        self.camera_source_var.set(
            self._camera_label_for_index(session.index)
        )
        self.demo_items = None
        self.result = None
        self.selected_item = None
        self.last_warp = None
        self.last_analysis_at = 0.0
        self.last_successful_analysis_at = 0.0
        self.camera_read_failures = 0
        self.camera_blank_frames = 0
        self.camera_blank_reconnects = 0
        self._clear_frame_candidates()
        self.stability.reset()
        self.screen_tracker.reset()
        self.screen_miss_count = 0
        self.camera_button.configure(text="웹캠 중지")
        self.view_mode.set("source")
        if (
            session.initial_frame is not None
            and not frame_is_blank(session.initial_frame)
        ):
            self.current_frame = session.initial_frame.copy()
            self.current_frame_id += 1
            self._show_image(self.current_frame)
        connected_name = (
            "Camo/iPhone 연결됨 · 프레임 수신 중"
            if session.is_camo
            else f"카메라 {session.index} 연결됨 · 프레임 수신 중"
        )
        self._set_connection_status(connected_name, "connected")
        self.camera_status_var.set(
            f"{session.description} 연결 완료 · 키오스크 화면을 카메라 안에 넣으세요."
        )
        self.status_var.set(
            "손떨림 중에도 선명한 프레임을 모으면 OCR이 자동으로 시작됩니다."
        )
        spoken_name = "아이폰 카메라" if session.is_camo else f"{session.index}번 카메라"
        self._speak_throttled(f"{spoken_name}가 연결되었습니다.", force=True)
        self._camera_tick(generation)

    def stop_camera(self, announce: bool = False) -> None:
        was_active = self.capture is not None or self.camera_open_future is not None
        was_camo = self.current_camera_is_camo
        self.camera_generation += 1
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.camera_open_future = None
        self.current_camera_is_camo = False
        self.camera_read_failures = 0
        self.camera_blank_frames = 0
        if announce:
            self.camera_blank_reconnects = 0
        self._clear_frame_candidates()
        self.stability.reset()
        self.screen_tracker.reset()
        self.screen_miss_count = 0
        if hasattr(self, "camera_button"):
            self.camera_button.configure(text="웹캠 시작")
        if hasattr(self, "camera_selector"):
            self.camera_selector.configure(state="readonly")
        if announce and was_active:
            disconnected = (
                "Camo/iPhone 연결 해제"
                if was_camo
                else "카메라 연결 해제"
            )
            self._set_connection_status(disconnected, "idle")
            self.camera_status_var.set("웹캠을 중지했습니다.")
            self._reset_tracking_feedback()
            self._speak_throttled("웹캠을 중지했습니다.", force=True)
        elif was_active and hasattr(self, "connection_status_var"):
            self._set_connection_status(
                self._idle_connection_message(), "idle"
            )

    def _camera_tick(self, generation: int) -> None:
        if self.capture is None or generation != self.camera_generation:
            return
        ok, frame = self.capture.read()
        if not ok or frame is None or not frame.size:
            self.camera_read_failures += 1
            if self.camera_read_failures < 5:
                source_name = (
                    "Camo/iPhone"
                    if self.current_camera_is_camo
                    else "카메라"
                )
                self._set_connection_status(
                    f"{source_name} 연결 불안정 · 프레임 재시도 "
                    f"{self.camera_read_failures}/5",
                    "warning",
                )
                self.camera_status_var.set(
                    f"웹캠 프레임 재시도 중 ({self.camera_read_failures}/5)..."
                )
                self.root.after(120, lambda: self._camera_tick(generation))
                return
            failed_camo = self.current_camera_is_camo
            self.stop_camera()
            failed_name = "Camo/iPhone" if failed_camo else "카메라"
            self._set_connection_status(
                f"{failed_name} 연결 끊김", "error"
            )
            self.camera_status_var.set(
                "웹캠 프레임을 5회 연속 읽지 못해 중지했습니다."
            )
            self._speak_throttled("웹캠 영상을 읽지 못해 중지했습니다.", force=True)
            return
        if self.current_camera_is_camo and frame_is_blank(frame):
            self.camera_blank_frames += 1
            self._set_connection_status(
                "Camo/iPhone 연결됨 · 영상 신호 대기 중",
                "warning",
            )
            self.camera_status_var.set(
                "검은 초기 프레임은 표시하지 않고 정상 영상을 기다리는 중 "
                f"({self.camera_blank_frames}/45)..."
            )
            if (
                self.camera_blank_frames >= 45
                and self.camera_blank_reconnects < 1
            ):
                self.camera_blank_reconnects += 1
                self.stop_camera()
                self._set_connection_status(
                    "Camo 검은 영상 감지 · 기본 포맷으로 다시 연결 중",
                    "warning",
                )
                self.root.after(450, self.toggle_camera)
                return
            if self.camera_blank_frames >= 45:
                self._set_connection_status(
                    "Camo는 연결됐지만 영상이 검습니다",
                    "error",
                )
                self.camera_status_var.set(
                    "Camo Studio 미리보기와 아이폰 후면 카메라를 확인한 뒤 "
                    "웹캠을 다시 시작하세요."
                )
            self.root.after(70, lambda: self._camera_tick(generation))
            return
        self.camera_read_failures = 0
        self.camera_blank_frames = 0
        self.camera_blank_reconnects = 0
        connected_name = (
            "Camo/iPhone 연결됨 · 프레임 수신 중"
            if self.current_camera_is_camo
            else "카메라 연결됨 · 프레임 수신 중"
        )
        expected_connection = f"● {connected_name}"
        if self.connection_status_var.get() != expected_connection:
            self._set_connection_status(connected_name, "connected")
        self.current_frame = frame
        self.current_frame_id += 1
        raw_detection = detect_screen(frame)
        using_full_frame = self.use_full_frame_tracking
        if using_full_frame:
            detection = full_frame_detection(frame)
            detection_source = "full_frame"
            tracking_confidence = 1.0
            self.screen_miss_count = 0
        else:
            detection, detection_source, tracking_confidence = (
                self.screen_tracker.update(frame, raw_detection)
            )
            if raw_detection is None:
                self.screen_miss_count += 1
            else:
                self.screen_miss_count = 0
        self.current_detection = detection
        if detection is None:
            message = (
                "흔들림으로 화면 테두리를 잠시 놓쳤습니다. "
                f"최근 선명 프레임 {len(self.frame_candidates)}장은 유지합니다."
            )
            self.camera_status_var.set(f"{self.camera_description} · {message}")
            self._speak_throttled(message, repeat_after=6.0)
            self.live_source_annotated = frame
            self.live_corrected_annotated = None
            self._show_image(frame)
            if self.screen_miss_count >= 10:
                self.stability.reset()
                self.screen_tracker.reset()
                self._clear_frame_candidates()
                self.camera_status_var.set(
                    f"{self.camera_description} · 화면을 오래 찾지 못했습니다. "
                    "네 모서리가 보이도록 다시 비추거나 '전체 프레임 OCR'을 누르세요."
                )
            else:
                self._maybe_start_camera_analysis(
                    time.time(),
                    stable=False,
                    movement=None,
                    force_full_frame=using_full_frame,
                )
            self._reset_tracking_feedback(keep_target=True)
            if self.selected_item is not None:
                self.target_status_var.set(
                    f"{self.selected_item.text} 목표 유지 · 키오스크 화면을 다시 찾는 중입니다."
                )
        else:
            message = (
                "전체 프레임 기준 추적"
                if using_full_frame
                else alignment_message(frame.shape, detection)
            )
            stable, movement = self.stability.update(detection.corners, frame.shape)
            annotated = draw_detection(frame, detection, message)
            current_warp = warp_screen(frame, detection.corners)
            now = time.time()
            candidate_state, frame_quality = self._record_frame_candidate(
                frame,
                detection,
                current_warp,
                now,
                movement,
                stable,
                detection_source,
                tracking_confidence,
                using_full_frame,
            )
            corrected = (
                annotate_warped(
                    current_warp,
                    self.result.items,
                    selected=self.selected_item,
                )
                if self.result is not None
                else current_warp
            )

            if self.selected_item is not None and self.result is not None:
                if self.manual_pointer is not None:
                    _, inside, _ = vibration_interval_ms(
                        self.manual_pointer,
                        self.selected_item,
                        self.analyzer.output_size,
                    )
                    corrected = draw_pointer_to_target(
                        corrected,
                        self.manual_pointer,
                        self.selected_item,
                        inside,
                    )
                    self.approach.reset()
                    self.tracking_was_inside = inside
                    self._update_manual_pointer_guidance()
                else:
                    tracking = track_camera_center(
                        frame.shape,
                        detection.corners,
                        self.selected_item,
                        self.analyzer.output_size,
                    )
                    rate = self.approach.update(tracking.normalized_distance)
                    annotated = draw_source_tracking(annotated, tracking)
                    corrected = draw_warped_pointer(
                        corrected, tracking, self.selected_item
                    )
                    if tracking.inside:
                        rate_message = "목표 중심 도착"
                    elif rate > 0.008:
                        rate_message = f"접근 중 +{rate:.1%}/초"
                    elif rate < -0.008:
                        rate_message = f"멀어지는 중 {rate:.1%}/초"
                    else:
                        rate_message = "거리 유지"
                    tracking_message = (
                        f"{self.selected_item.text} · {tracking.direction} · "
                        f"정규화 거리 {tracking.normalized_distance:.1%} · "
                        f"{rate_message}"
                    )
                    self.target_status_var.set(tracking_message)
                    self._set_haptic_feedback(
                        active=True,
                        interval=tracking.interval_ms,
                        inside=tracking.inside,
                        detail=tracking_message,
                    )
                    relative_location = target_direction_label(
                        tracking.pointer_screen, self.selected_item
                    )
                    voice_message = (
                        f"{self.selected_item.text}, "
                        f"{relative_location}에 있습니다."
                    )
                    self.location_voice_message = voice_message
                    self.tracking_was_inside = tracking.inside
            else:
                self._reset_tracking_feedback(keep_target=True)

            self.live_source_annotated = annotated
            self.live_corrected_annotated = corrected
            if self.view_mode.get() == "source" or self.result is None:
                self._show_image(annotated)
            else:
                self._show_image(corrected)

            if using_full_frame:
                camera_state = "전체 프레임을 키오스크 좌표로 사용 중"
            elif detection_source == "tracked":
                camera_state = (
                    "테두리를 잠시 놓쳤지만 광학 추적으로 화면을 유지 중 · "
                    f"선명 후보 {len(self.frame_candidates)}장"
                )
            elif candidate_state == "clipped":
                camera_state = (
                    "화면 모서리가 잘렸습니다 · 네 모서리를 카메라 안쪽에 넣으세요"
                )
            elif candidate_state == "blurry":
                camera_state = (
                    f"손떨림으로 글자가 흐림(품질 {frame_quality:.1f}) · "
                    "선명한 순간을 자동으로 기다리는 중"
                )
            elif candidate_state == "shaking":
                camera_state = (
                    "손떨림 보정 중 · 조금만 천천히 움직이면 자동 인식합니다"
                )
            elif not stable:
                camera_state = (
                    f"손떨림 허용 모드 · 선명 후보 "
                    f"{len(self.frame_candidates)}장 수집"
                )
            elif message == "화면 정렬 완료":
                camera_state = (
                    f"화면 인식 완료 · 선명 후보 "
                    f"{len(self.frame_candidates)}장 중 최적 장면 선택"
                )
            else:
                camera_state = (
                    f"{message} · 선명 후보 {len(self.frame_candidates)}장"
                )
            self.camera_status_var.set(
                f"{self.camera_description} · {camera_state}"
            )
            if self.selected_item is None and not self.analysis_in_progress:
                self._speak_throttled(
                    f"카메라 안내. {camera_state}", repeat_after=6.0
                )

            self._maybe_start_camera_analysis(
                now,
                stable,
                movement,
                using_full_frame,
            )
        self.root.after(70, lambda: self._camera_tick(generation))

    def analyze_current(
        self,
        auto: bool = False,
        force_full_frame: bool = False,
        detection=None,
        comparison_warp=None,
        announce_coordinate: bool = False,
    ) -> None:
        if self.current_frame is None:
            self._speak_throttled("분석할 화면이 없습니다.", force=True)
            return
        self.last_analysis_at = time.time()
        frame = self.current_frame.copy()
        if self.capture is not None and self.frame_candidates:
            best_candidate = select_best_ocr_candidate(
                self.frame_candidates,
                time.time(),
                max_age=3.0,
            )
            if best_candidate is not None:
                frame = best_candidate["frame"].copy()
                if not force_full_frame:
                    detection = best_candidate["detection"]
                    comparison_warp = best_candidate["warp"].copy()
        selected_detection = detection
        if selected_detection is None:
            selected_detection = (
                detect_screen(frame)
                if self.use_full_frame_tracking and not force_full_frame
                else self.current_detection
            )
        if force_full_frame:
            selected_detection = full_frame_detection(frame)
            comparison_warp = warp_screen(
                frame, selected_detection.corners, self.analyzer.output_size
            )
        if comparison_warp is None and selected_detection is not None:
            comparison_warp = warp_screen(
                frame, selected_detection.corners, self.analyzer.output_size
            )
        request = {
            "frame": frame,
            "demo_items": self.demo_items,
            "auto": auto,
            "force_full_frame": force_full_frame,
            "announce_coordinate": announce_coordinate,
            "detection": selected_detection,
            "comparison_warp": (
                comparison_warp.copy() if comparison_warp is not None else None
            ),
            "previous": self.selected_item,
            "old_signature": self._result_signature(self.result),
            "list_had_focus": self.root.focus_get() is self.item_list,
            "input_generation": self.input_generation,
            "frame_id": self.current_frame_id,
            "screen_epoch": self.screen_epoch,
        }
        if self.analysis_in_progress:
            if (
                auto
                and self.pending_analysis is not None
                and not self.pending_analysis["auto"]
            ):
                return
            self.analysis_request_id += 1
            request["request_id"] = self.analysis_request_id
            self.pending_analysis = request
            if not auto:
                self.status_var.set("OCR 분석 중입니다. 최신 화면을 이어서 분석합니다.")
            return
        self.analysis_request_id += 1
        request["request_id"] = self.analysis_request_id
        self._start_analysis(request)

    def _start_analysis(self, request) -> None:
        self.analysis_in_progress = True
        self.analyze_button.configure(state="disabled")
        self.full_frame_button.configure(state="disabled")
        mode = "전체 프레임" if request["force_full_frame"] else "검출 화면"
        self.status_var.set(f"{mode} 보정 및 OCR 분석 중… 카메라는 계속 작동합니다.")
        future = self.analysis_executor.submit(
            self._run_analysis_job,
            request["frame"],
            request["demo_items"],
            request["detection"],
        )
        self.analysis_future = future
        self.root.after(80, lambda: self._poll_analysis(future, request))

    def _run_analysis_job(self, frame, demo_items, detection):
        fallback_note = ""
        try:
            override = None
            if demo_items is not None and not self.ocr.available:
                override = StaticOCR(demo_items)
            result = self.analyzer.analyze(frame, override, detection)
        except Exception as error:
            if demo_items is not None:
                try:
                    result = self.analyzer.analyze(
                        frame, StaticOCR(demo_items), detection
                    )
                    fallback_note = f"실제 OCR 실패로 데모 좌표 사용: {error}"
                except Exception as fallback_error:
                    raise RuntimeError(str(fallback_error)) from fallback_error
            else:
                raise
        return result, fallback_note

    def _poll_analysis(self, future: Future, request) -> None:
        if self.closed:
            return
        if not future.done():
            self.root.after(80, lambda: self._poll_analysis(future, request))
            return
        self.analysis_future = None
        self.analysis_in_progress = False
        self.analyze_button.configure(state="normal")
        self.full_frame_button.configure(state="normal")
        is_current = (
            request["input_generation"] == self.input_generation
            and request["request_id"] == self.analysis_request_id
            and request["screen_epoch"] == self.screen_epoch
        )
        try:
            result, fallback_note = future.result()
        except Exception as error:
            if is_current:
                hint = (
                    " 화면 테두리가 보이지 않으면 '전체 프레임 OCR'을 사용하세요."
                    if "모서리" in str(error)
                    else ""
                )
                self.status_var.set(f"{error}{hint}")
                self._speak_throttled(f"분석에 실패했습니다. {error}", force=True)
                if not request["auto"]:
                    messagebox.showwarning("분석 실패", f"{error}{hint}")
        else:
            if is_current:
                self._apply_analysis_result(result, fallback_note, request)
        if self.pending_analysis is not None:
            pending = self.pending_analysis
            self.pending_analysis = None
            self._start_analysis(pending)

    def _apply_analysis_result(self, result, fallback_note: str, request) -> None:
        previous = request["previous"]
        old_signature = request["old_signature"]
        new_signature = self._result_signature(result)
        changed = new_signature != old_signature
        self.result = result
        self.pending_candidate_index = None
        self.use_full_frame_tracking = (
            request["force_full_frame"] and self.capture is not None
        )
        self.item_list.delete(0, tk.END)
        for item in result.items:
            confidence_label = self._confidence_label(item.confidence)
            self.item_list.insert(
                tk.END,
                (
                    f"{item.text}  |  {item.grid} {item.relative} · "
                    f"{confidence_label} {item.confidence:.0f}%"
                ).strip(),
            )
        mode = "전체 프레임" if request["force_full_frame"] else "화면 검출"
        reliable_count = sum(item.confidence >= 70 for item in result.items)
        review_count = len(result.items) - reliable_count
        status = (
            f"{mode} · 확실 {reliable_count}개"
            + (f" · 확인 필요 {review_count}개" if review_count else "")
            + f" · {result.ocr_engine}"
        )
        hints = result.ocr_diagnostics.get("hints", [])
        if hints:
            status = f"{status} · {hints[0]}"
        if fallback_note:
            status = f"{status} · {fallback_note}"
        if result.items:
            self.last_warp = (
                request["comparison_warp"].copy()
                if request["comparison_warp"] is not None
                else result.warped.copy()
            )
            self.last_successful_analysis_at = time.time()
            self.view_mode.set("corrected")
            self.status_var.set(status)
        else:
            self.status_var.set(
                f"{status} · 글자를 찾지 못했습니다. 화면을 가까이 하고 다시 시도합니다."
            )

        query = self.target_query_var.get().strip()
        if previous is not None:
            match_index = find_same_text_item_index(previous, result.items)
            if (
                match_index is not None
                and result.items[match_index].confidence < 45
            ):
                match_index = None
            if match_index is None:
                self.target_status_var.set(
                    f"기존 목표 '{previous.text}'가 화면에서 사라졌습니다. 다시 선택하세요."
                )
        elif query:
            query_match = find_query_match(query, result.items)
            if query_match is not None:
                index, score = query_match
                candidate = result.items[index]
                if score >= 0.98 and candidate.confidence >= 55:
                    match_index = index
                    self.target_status_var.set(
                        f"입력한 '{query}' 메뉴를 높은 신뢰도로 찾았습니다."
                    )
                else:
                    match_index = None
                    self._highlight_candidate(index)
                    self.target_status_var.set(
                        f"후보: '{candidate.text}' · 검색 유사도 {score:.0%} · "
                        f"OCR {candidate.confidence:.0f}%. 목록에서 클릭해 확인하세요."
                    )
            else:
                match_index = None
                self.target_status_var.set(
                    f"'{query}' 메뉴를 찾지 못했습니다. 글자 인식 결과를 확인하세요."
                )
        elif not request["auto"] and result.items:
            match_index = 0
        else:
            match_index = None
        if match_index is not None:
            self._activate_item(
                match_index,
                speak=bool(request.get("announce_coordinate", False)),
                preserve_pointer=(
                    previous is not None
                    and self.manual_pointer is not None
                ),
            )
            if request["list_had_focus"] or not request["auto"]:
                self.item_list.focus_set()
        else:
            self.selected_item = None
            self.manual_pointer = None
            self.location_voice_message = ""
            self._reset_tracking_feedback(keep_target=True)
            if not query and previous is None:
                self.target_status_var.set(
                    "메뉴명을 입력하거나 아래 목록에서 목표를 선택하세요."
                )
            self.guide_var.set(
                "인식된 메뉴가 없습니다."
                if not result.items
                else "찾을 메뉴를 입력하거나 목록에서 목표를 선택하세요."
            )
            self._render_current()

        if request["auto"]:
            if changed and self.location_voice_message:
                self._speak_throttled(self.location_voice_message, force=True)
        else:
            if self.location_voice_message:
                self._speak_throttled(self.location_voice_message, force=True)

    @staticmethod
    def _result_signature(result) -> tuple:
        if result is None:
            return ()
        return tuple(
            (
                item.text,
                item.grid,
                item.relative,
                item.x,
                item.y,
                item.width,
                item.height,
                round(item.confidence, 1),
            )
            for item in result.items
        )

    @staticmethod
    def _confidence_label(confidence: float) -> str:
        if confidence >= 70:
            return "높음"
        if confidence >= 50:
            return "확인 필요"
        return "낮음"

    def _highlight_candidate(self, index: int) -> None:
        if not self.result or not (0 <= index < len(self.result.items)):
            return
        self.pending_candidate_index = index
        self.item_list.selection_clear(0, tk.END)
        self.item_list.selection_set(index)
        self.item_list.activate(index)
        self.item_list.see(index)

    def _handle_search_match(
        self,
        query: str,
        match: tuple[int, float],
        speak: bool,
    ) -> None:
        if self.result is None:
            return
        index, score = match
        candidate = self.result.items[index]
        if score >= 0.98 and candidate.confidence >= 55:
            self._activate_item(index, speak=speak)
            self.item_list.focus_set()
            return
        self.selected_item = None
        self.manual_pointer = None
        self.location_voice_message = ""
        self._reset_tracking_feedback(keep_target=True)
        self._highlight_candidate(index)
        message = (
            f"검색어 '{query}'의 후보는 '{candidate.text}'입니다. "
            f"유사도 {score:.0%}, OCR 신뢰도 {candidate.confidence:.0f}퍼센트입니다. "
            "목록에서 클릭하거나 엔터를 눌러 확인하세요."
        )
        self.target_status_var.set(message)
        self.guide_var.set("후보 확인 전에는 목표 추적과 진동을 시작하지 않습니다.")
        self._render_current()
        if speak:
            self._speak_throttled(
                f"후보는 {candidate.text}입니다. 목록에서 확인하세요.",
                force=True,
            )

    def search_target(self, event=None) -> None:
        query = self.target_query_var.get().strip()
        if not query:
            self.target_status_var.set("찾을 메뉴명을 먼저 입력하세요.")
            self._speak_throttled("찾을 메뉴명을 먼저 입력하세요.", force=True)
            self.target_entry.focus_set()
            return
        if self.result is not None and self.result.items:
            match = find_query_match(query, self.result.items)
            if match is not None:
                self._handle_search_match(query, match, speak=True)
                return
        if self.current_frame is None:
            self.target_status_var.set(
                f"'{query}'을 찾으려면 먼저 카메라나 사진을 불러오세요."
            )
            return
        force_full_frame = self.capture is not None and self.current_detection is None
        self.selected_item = None
        self.pending_candidate_index = None
        self.manual_pointer = None
        self.location_voice_message = ""
        self._reset_tracking_feedback(keep_target=True)
        self.target_status_var.set(
            f"최신 화면에서 '{query}' 메뉴를 OCR로 찾는 중입니다."
        )
        self.analyze_current(
            auto=False,
            force_full_frame=force_full_frame,
            detection=self.current_detection,
            announce_coordinate=True,
        )

    def select_item(self, event=None) -> None:
        if not self.result or not self.result.items:
            return
        selection = self.item_list.curselection()
        if not selection:
            return
        self._activate_item(selection[0], speak=True)

    def _activate_item(
        self,
        index: int,
        speak: bool,
        preserve_pointer: bool = False,
    ) -> None:
        if not self.result or not (0 <= index < len(self.result.items)):
            return
        preserved_pointer = self.manual_pointer if preserve_pointer else None
        self.item_list.selection_clear(0, tk.END)
        self.item_list.selection_set(index)
        self.item_list.activate(index)
        self.item_list.see(index)
        self.selected_item = self.result.items[index]
        self.pending_candidate_index = None
        self.manual_pointer = preserved_pointer
        self.target_query_var.set(self.selected_item.text)
        self.approach.reset()
        self.tracking_was_inside = False
        relative = f", {self.selected_item.relative}" if self.selected_item.relative else ""
        screen_height, screen_width = self.result.warped.shape[:2]
        position = screen_position_label(
            self.selected_item, (screen_width, screen_height)
        )
        grid_x, grid_y = grid_numeric_coordinate(self.selected_item)
        if self.manual_pointer is not None:
            self._update_manual_pointer_guidance()
            voice_message = self.location_voice_message
        else:
            display_message = (
                f"{self.selected_item.text}, 화면 {position}에 있습니다. "
                f"촉각 격자 {self.selected_item.grid}{relative} · "
                f"좌표 ({grid_x}, {grid_y}), "
                f"OCR 신뢰도 {self.selected_item.confidence:.0f}퍼센트입니다."
            )
            voice_message = (
                f"{self.selected_item.text}, 화면 {position}에 있습니다."
            )
            self.location_voice_message = voice_message
            self.guide_var.set(display_message)
            if self.capture is not None:
                self.target_status_var.set(
                    f"{self.selected_item.text} 목표 확정 · "
                    "화면을 클릭하면 그 위치를 기준으로 다시 안내합니다."
                )
            else:
                self.target_status_var.set(
                    f"{self.selected_item.text} 목표 확정 · "
                    "보정 화면에서 위치를 확인하세요."
                )
        self._render_current()
        if speak:
            self._announce_selected_coordinate()

    def _announce_selected_coordinate(self) -> None:
        if self.selected_item is None:
            return
        grid_x, grid_y = grid_numeric_coordinate(self.selected_item)
        shown_message = (
            f"현재 {self.selected_item.text}는 ({grid_x}, {grid_y})에 있습니다."
        )
        spoken_message = (
            f"현재 {self.selected_item.text}는 좌표 "
            f"{grid_x}, {grid_y}에 있습니다."
        )
        next_action = (
            "화면을 클릭하면 그 위치를 기준으로 다시 안내합니다."
            if self.capture is not None
            else "보정 화면을 클릭하면 그 위치를 기준으로 다시 안내합니다."
        )
        self.target_status_var.set(f"{shown_message} · {next_action}")
        self.status_var.set("선택한 메뉴의 격자 좌표를 안내합니다.")
        if self.tts_var.get():
            self.speaker.speak(spoken_message, interrupt=True)
            self.last_spoken = spoken_message
            self.last_spoken_at = time.time()

    def _speak_throttled(
        self,
        message: str,
        force: bool = False,
        repeat_after: float = 3.0,
        manual: bool = False,
    ) -> None:
        if not manual:
            return
        now = time.time()
        if force or message != self.last_spoken or now - self.last_spoken_at > repeat_after:
            self.speaker.speak(message, interrupt=force)
            self.last_spoken = message
            self.last_spoken_at = now

    def repeat_current_guidance(self, event=None) -> None:
        if not self.tts_var.get():
            self.status_var.set(
                "음성 사용을 켠 뒤 R 또는 '안내 듣기' 버튼을 누르세요."
            )
            return
        if self.manual_pointer is not None and self.selected_item is not None:
            self._update_manual_pointer_guidance()
        if not self.location_voice_message:
            self.status_var.set(
                "먼저 메뉴를 선택하세요. 위치가 계산되면 R로 들을 수 있습니다."
            )
            return
        self.status_var.set(
            "클릭 위치 기준 안내를 재생합니다."
            if self.manual_pointer is not None
            else "현재 위치 안내를 재생합니다."
        )
        self._speak_throttled(
            self.location_voice_message,
            force=True,
            manual=True,
        )

    def _render_current(self) -> None:
        if not self.canvas.winfo_exists():
            return
        if self.capture is not None:
            if self.view_mode.get() == "source":
                image = (
                    self.live_source_annotated
                    if self.live_source_annotated is not None
                    else self.current_frame
                )
            else:
                if self.live_corrected_annotated is not None:
                    image = self.live_corrected_annotated
                elif self.live_source_annotated is not None:
                    image = self.live_source_annotated
                else:
                    image = self.current_frame
        elif self.result is not None:
            if self.view_mode.get() == "source":
                image = self.result.source_annotated
            else:
                image = annotate_warped(
                    self.result.warped,
                    self.result.items,
                    selected=self.selected_item,
                )
        else:
            image = self.current_frame
        if (
            image is not None
            and self.view_mode.get() == "corrected"
            and self.manual_pointer is not None
            and self.selected_item is not None
        ):
            _, inside, _ = vibration_interval_ms(
                self.manual_pointer,
                self.selected_item,
                self.analyzer.output_size,
            )
            image = draw_pointer_to_target(
                image,
                self.manual_pointer,
                self.selected_item,
                inside,
            )
        if image is not None:
            self._show_image(image)

    def _show_image(self, image) -> None:
        canvas_width = max(10, self.canvas.winfo_width())
        canvas_height = max(10, self.canvas.winfo_height())
        height, width = image.shape[:2]
        scale = min(canvas_width / width, canvas_height / height)
        shown_width = max(1, int(width * scale))
        shown_height = max(1, int(height * scale))
        resized = cv2.resize(image, (shown_width, shown_height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        x_offset = (canvas_width - shown_width) // 2
        y_offset = (canvas_height - shown_height) // 2
        self.canvas.delete("all")
        self.canvas.create_image(x_offset, y_offset, anchor="nw", image=self.photo)
        self.display_geometry = (
            x_offset,
            y_offset,
            shown_width,
            shown_height,
            width,
            height,
        )

    def _pointer_motion(self, event) -> None:
        if (
            self.capture is not None
            or
            self.view_mode.get() != "corrected"
            or self.selected_item is None
            or self.display_geometry is None
        ):
            return
        x0, y0, shown_w, shown_h, image_w, image_h = self.display_geometry
        if not (x0 <= event.x <= x0 + shown_w and y0 <= event.y <= y0 + shown_h):
            return
        px = (event.x - x0) / shown_w * image_w
        py = (event.y - y0) / shown_h * image_h
        interval, inside, normalized = vibration_interval_ms(
            (px, py), self.selected_item, (image_w, image_h)
        )
        if inside:
            detail = "목표 안쪽 · 강한 연속 진동"
        else:
            detail = f"목표 거리 {normalized:.1%} · 진동 간격 약 {interval} ms"
        self._set_haptic_feedback(
            active=True,
            interval=interval,
            inside=inside,
            detail=detail,
        )

    def _update_manual_pointer_guidance(
        self,
        render: bool = False,
    ) -> tuple[int, bool, float] | None:
        if self.manual_pointer is None or self.selected_item is None:
            return None
        direction = target_direction_label(
            self.manual_pointer,
            self.selected_item,
        )
        voice_message = (
            f"{self.selected_item.text}, {direction}에 있습니다."
        )
        self.location_voice_message = voice_message
        self.target_status_var.set(
            f"클릭 위치 기준 재분석 · {voice_message} · R로 듣기"
        )
        self.guide_var.set(voice_message)
        interval, inside, normalized = vibration_interval_ms(
            self.manual_pointer,
            self.selected_item,
            self.analyzer.output_size,
        )
        detail = (
            "클릭 위치가 목표 안쪽 · 강한 연속 진동"
            if inside
            else (
                f"클릭 위치 기준 거리 {normalized:.1%} · "
                f"진동 간격 약 {interval} ms"
            )
        )
        self._set_haptic_feedback(
            active=True,
            interval=interval,
            inside=inside,
            detail=detail,
        )
        if render:
            self._render_current()
        return interval, inside, normalized

    def _canvas_click(self, event) -> None:
        if (
            self.result is None
            or self.selected_item is None
            or self.display_geometry is None
        ):
            return
        x0, y0, shown_w, shown_h, image_w, image_h = self.display_geometry
        if not (x0 <= event.x <= x0 + shown_w and y0 <= event.y <= y0 + shown_h):
            return
        image_point = (
            (event.x - x0) / shown_w * image_w,
            (event.y - y0) / shown_h * image_h,
        )
        if self.view_mode.get() == "corrected":
            if self.capture is not None and self.current_detection is None:
                self.target_status_var.set(
                    "키오스크 화면을 다시 찾은 뒤 클릭 위치를 분석할 수 있습니다."
                )
                return
            pointer = image_point
        else:
            detection = (
                self.current_detection
                if self.capture is not None
                else self.result.detection
            )
            if detection is None:
                return
            pointer = frame_point_to_screen(
                image_point,
                detection.corners,
                self.analyzer.output_size,
            )

        self.manual_pointer = pointer
        self._update_manual_pointer_guidance(render=True)

    def _set_haptic_feedback(
        self,
        active: bool,
        interval: int = 1000,
        inside: bool = False,
        detail: str = "",
    ) -> None:
        self.haptic_active = active
        self.haptic_interval = max(80, int(interval))
        self.haptic_inside = inside
        self.haptic_detail = detail
        if detail:
            self.haptic_var.set(detail)

    def _reset_tracking_feedback(self, keep_target: bool = False) -> None:
        self.approach.reset()
        self.tracking_was_inside = False
        self.haptic_active = False
        self.haptic_inside = False
        self.haptic_interval = 1000
        self.haptic_detail = "목표를 선택하면 진동이 시작됩니다."
        if hasattr(self, "haptic_var"):
            self.haptic_var.set(self.haptic_detail)
        if not keep_target and hasattr(self, "target_status_var"):
            self.manual_pointer = None
            self.location_voice_message = ""
            self.target_status_var.set(
                "메뉴명을 입력하거나 아래 목록에서 목표를 선택하세요."
            )

    def _haptic_tick(self) -> None:
        if self.closed or not self.haptic_panel.winfo_exists():
            return
        if not self.haptic_active:
            self.haptic_panel.configure(
                text="진동 대기", background="#E9EEF2", foreground="#34495E"
            )
        elif self.haptic_inside:
            self.haptic_panel.configure(
                text="목표 일치 — 강한 연속 진동",
                background="#38B66B",
                foreground="#FFFFFF",
            )
        else:
            phase = int(time.monotonic() * 1000) % self.haptic_interval
            pulse_on = phase < min(140, self.haptic_interval // 2)
            self.haptic_panel.configure(
                text=f"가상 진동 · {self.haptic_interval} ms 간격",
                background="#F2B544" if pulse_on else "#FFF2CB",
                foreground="#17202A",
            )
        self.root.after(70, self._haptic_tick)

    def close(self) -> None:
        self.closed = True
        self.stop_camera()
        self.speaker.stop()
        self.camera_executor.shutdown(wait=False, cancel_futures=True)
        self.analysis_executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def launch(project_root: Path) -> None:
    root = tk.Tk()
    PrototypeApp(root, project_root)
    root.update_idletasks()
    root.deiconify()
    root.lift()
    try:
        root.attributes("-topmost", True)
        root.after(450, lambda: root.attributes("-topmost", False))
    except tk.TclError:
        pass
    root.mainloop()
