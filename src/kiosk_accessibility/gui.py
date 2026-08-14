from __future__ import annotations

import time
import tkinter as tk
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from statistics import median
from tkinter import filedialog, font as tkfont, messagebox, ttk

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
        self.haptic_detail = "목표를 선택하면 가상 진동 안내가 시작됩니다."
        self.last_spoken = ""
        self.last_spoken_at = 0.0
        self.photo = None
        self.display_geometry = None
        self._result_selection_guard = False

        self.root.title("키오스크 메뉴 길찾기")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = max(1120, min(1440, screen_width - 80))
        window_height = max(720, min(920, screen_height - 100))
        left = max(0, (screen_width - window_width) // 2)
        top = max(0, (screen_height - window_height) // 2)
        self.root.geometry(
            f"{window_width}x{window_height}+{left}+{top}"
        )
        self.root.minsize(min(1120, window_width), min(720, window_height))
        self.root.configure(background="#F4F7FB")
        self._build()
        self.root.after(80, self._haptic_tick)
        self.load_demo()

    def _build(self) -> None:
        colors = {
            "background": "#F4F7FB",
            "surface": "#FFFFFF",
            "surface_soft": "#F7F9FC",
            "border": "#D9E1EC",
            "text": "#172033",
            "muted": "#5D6B7D",
            "navy": "#10233F",
            "navy_soft": "#183456",
            "accent": "#2457D6",
            "accent_dark": "#1946B7",
            "accent_soft": "#EAF0FF",
            "success": "#087A55",
            "success_soft": "#E8F7F1",
            "warning": "#9A5A08",
            "warning_soft": "#FFF3D6",
            "danger": "#B42318",
            "canvas": "#0E1726",
        }
        self.ui_colors = colors
        available_fonts = set(tkfont.families(self.root))
        font = next(
            (
                family
                for family in (
                    "Pretendard",
                    "SUIT",
                    "Noto Sans KR",
                    "맑은 고딕",
                )
                if family in available_fonts
            ),
            "맑은 고딕",
        )
        self.root.option_add("*Font", (font, 10))
        self.root.option_add("*TCombobox*Listbox.font", (font, 10))

        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", font=(font, 10))
        style.configure("App.TFrame", background=colors["background"])
        style.configure("Card.TFrame", background=colors["surface"])
        style.configure("Soft.TFrame", background=colors["surface_soft"])
        style.configure(
            "Card.TLabel",
            background=colors["surface"],
            foreground=colors["text"],
        )
        style.configure(
            "CardTitle.TLabel",
            background=colors["surface"],
            foreground=colors["text"],
            font=(font, 12, "bold"),
        )
        style.configure(
            "CardHint.TLabel",
            background=colors["surface"],
            foreground=colors["muted"],
            font=(font, 9),
        )
        style.configure(
            "Soft.TLabel",
            background=colors["surface_soft"],
            foreground=colors["muted"],
        )
        style.configure(
            "Primary.TButton",
            background=colors["accent"],
            foreground="#FFFFFF",
            borderwidth=0,
            focusthickness=2,
            focuscolor=colors["accent_dark"],
            padding=(13, 11),
            font=(font, 10, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[
                ("pressed", colors["accent_dark"]),
                ("active", "#2F67EA"),
                ("disabled", "#A9B7D2"),
            ],
            foreground=[("disabled", "#F5F7FA")],
        )
        style.configure(
            "Secondary.TButton",
            background="#F1F5F9",
            foreground=colors["text"],
            borderwidth=0,
            bordercolor="#F1F5F9",
            lightcolor="#F1F5F9",
            darkcolor="#F1F5F9",
            padding=(11, 10),
            font=(font, 10, "bold"),
        )
        style.map(
            "Secondary.TButton",
            background=[("pressed", "#DDE5EF"), ("active", "#E7EDF4")],
        )
        style.configure(
            "Tertiary.TButton",
            background=colors["surface"],
            foreground=colors["accent_dark"],
            borderwidth=0,
            padding=(8, 6),
            font=(font, 9, "bold"),
        )
        style.map(
            "Tertiary.TButton",
            background=[("pressed", colors["accent_soft"]), ("active", colors["accent_soft"])],
        )
        style.configure(
            "Search.TEntry",
            fieldbackground="#FFFFFF",
            foreground=colors["text"],
            bordercolor="#B8C5D8",
            lightcolor="#B8C5D8",
            darkcolor="#B8C5D8",
            padding=9,
            font=(font, 11),
        )
        style.configure(
            "Camera.TCombobox",
            fieldbackground="#FFFFFF",
            background="#FFFFFF",
            foreground=colors["text"],
            bordercolor="#B8C5D8",
            lightcolor="#B8C5D8",
            darkcolor="#B8C5D8",
            arrowcolor=colors["muted"],
            padding=8,
        )
        style.map(
            "Camera.TCombobox",
            fieldbackground=[("readonly", "#FFFFFF"), ("disabled", "#EEF2F7")],
            foreground=[("disabled", "#94A3B8")],
        )
        style.configure(
            "Card.TCheckbutton",
            background=colors["surface"],
            foreground=colors["text"],
            font=(font, 9),
        )
        style.map(
            "Card.TCheckbutton",
            background=[("active", colors["surface"])],
        )
        style.configure(
            "Analysis.Horizontal.TProgressbar",
            troughcolor="#E7EDF5",
            background=colors["accent"],
            lightcolor=colors["accent"],
            darkcolor=colors["accent"],
            bordercolor="#E7EDF5",
            thickness=4,
        )
        style.configure(
            "Result.Treeview",
            background="#FFFFFF",
            fieldbackground="#FFFFFF",
            foreground=colors["text"],
            borderwidth=0,
            relief="flat",
            rowheight=42,
            font=(font, 10),
        )
        style.map(
            "Result.Treeview",
            background=[("selected", colors["accent_soft"])],
            foreground=[("selected", colors["accent_dark"])],
        )
        style.configure(
            "Result.Treeview.Heading",
            background="#F8FAFC",
            foreground=colors["muted"],
            borderwidth=0,
            relief="flat",
            padding=(8, 8),
            font=(font, 9, "bold"),
        )
        style.map(
            "Result.Treeview.Heading",
            background=[("active", "#F1F5F9")],
        )
        style.configure(
            "Modern.Vertical.TScrollbar",
            background="#CBD5E1",
            troughcolor="#F8FAFC",
            bordercolor="#F8FAFC",
            lightcolor="#CBD5E1",
            darkcolor="#CBD5E1",
            arrowcolor="#64748B",
            width=10,
        )

        header = tk.Frame(
            self.root,
            background=colors["navy"],
            padx=24,
            pady=14,
        )
        header.pack(fill="x")
        title_block = tk.Frame(header, background=colors["navy"])
        title_block.pack(side="left", fill="x", expand=True)
        tk.Label(
            title_block,
            text="ACCESSIBLE KIOSK GUIDE",
            background=colors["navy"],
            foreground="#8EB5FF",
            font=(font, 9, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_block,
            text="키오스크 메뉴 길찾기",
            background=colors["navy"],
            foreground="#FFFFFF",
            font=(font, 21, "bold"),
        ).pack(anchor="w", pady=(1, 0))
        tk.Label(
            title_block,
            text="카메라 화면을 인식해 원하는 메뉴의 위치를 음성으로 안내합니다.",
            background=colors["navy"],
            foreground="#C7D4E7",
            font=(font, 9),
        ).pack(anchor="w", pady=(2, 0))

        engine_badge = tk.Frame(
            header,
            background=colors["navy_soft"],
            padx=13,
            pady=8,
        )
        engine_badge.pack(side="right", padx=(16, 0))
        tk.Label(
            engine_badge,
            text="OCR ENGINE",
            background=colors["navy_soft"],
            foreground="#8EB5FF",
            font=(font, 8, "bold"),
        ).pack(anchor="e")
        self.engine_label = tk.Label(
            engine_badge,
            text=self.ocr.name,
            background=colors["navy_soft"],
            foreground="#FFFFFF",
            font=(font, 9, "bold"),
        )
        self.engine_label.pack(anchor="e", pady=(1, 0))

        body = ttk.Frame(self.root, style="App.TFrame")
        body.pack(fill="both", expand=True, padx=18, pady=(16, 10))

        viewer = ttk.Frame(body, style="App.TFrame", padding=(0, 0, 7, 0))
        controls_shell = ttk.Frame(
            body,
            style="App.TFrame",
            padding=(7, 0, 0, 0),
            width=450,
        )
        controls_shell.pack(side="right", fill="y")
        controls_shell.pack_propagate(False)
        viewer.pack(side="left", fill="both", expand=True)

        viewer_card = tk.Frame(
            viewer,
            background=colors["surface"],
            highlightbackground=colors["border"],
            highlightthickness=1,
            bd=0,
        )
        viewer_card.pack(fill="both", expand=True)
        viewer_toolbar = tk.Frame(
            viewer_card, background=colors["surface"], padx=15, pady=11
        )
        viewer_toolbar.pack(fill="x")
        viewer_title = tk.Frame(viewer_toolbar, background=colors["surface"])
        viewer_title.pack(side="left")
        tk.Label(
            viewer_title,
            text="실시간 화면",
            background=colors["surface"],
            foreground=colors["text"],
            font=(font, 12, "bold"),
        ).pack(anchor="w")
        tk.Label(
            viewer_title,
            text="메뉴 선택 후 화면을 클릭하면 해당 위치를 기준으로 다시 계산합니다.",
            background=colors["surface"],
            foreground=colors["muted"],
            font=(font, 8),
        ).pack(anchor="w", pady=(1, 0))

        self.view_mode = tk.StringVar(value="corrected")
        mode_group = tk.Frame(
            viewer_toolbar,
            background="#EEF2F7",
            padx=3,
            pady=3,
        )
        mode_group.pack(side="right")
        for label, value in (
            ("카메라 원본", "source"),
            ("보정된 화면", "corrected"),
        ):
            tk.Radiobutton(
                mode_group,
                text=label,
                variable=self.view_mode,
                value=value,
                command=self._render_current,
                indicatoron=False,
                borderwidth=0,
                relief="flat",
                background="#EEF2F7",
                activebackground="#E2E8F0",
                selectcolor="#FFFFFF",
                foreground=colors["text"],
                activeforeground=colors["accent_dark"],
                font=(font, 9, "bold"),
                padx=12,
                pady=7,
                cursor="hand2",
            ).pack(side="left")

        self.haptic_var = tk.StringVar(
            value="목표를 선택하면 가상 진동 안내가 시작됩니다."
        )
        self.canvas = tk.Canvas(
            viewer_card,
            background=colors["canvas"],
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True, padx=12)
        self.canvas.bind("<Configure>", lambda event: self._render_current())
        self.canvas.bind("<Motion>", self._pointer_motion)
        self.canvas.bind("<Button-1>", self._canvas_click)
        self.canvas.bind(
            "<Leave>",
            lambda event: self.haptic_var.set(
                "포인터를 보정된 화면 위로 이동하세요."
            ),
        )

        viewer_status = tk.Frame(
            viewer_card,
            background=colors["surface_soft"],
            padx=14,
            pady=9,
        )
        viewer_status.pack(fill="x", padx=12, pady=(0, 12))
        tk.Label(
            viewer_status,
            textvariable=self.haptic_var,
            background=colors["surface_soft"],
            foreground=colors["muted"],
            font=(font, 9, "bold"),
            anchor="w",
        ).pack(side="left", fill="x", expand=True)
        legend = tk.Frame(viewer_status, background=colors["surface_soft"])
        legend.pack(side="right", padx=(12, 0))
        tk.Label(
            legend,
            text="●",
            background=colors["surface_soft"],
            foreground=colors["success"],
            font=(font, 8, "bold"),
        ).pack(side="left")
        tk.Label(
            legend,
            text="신뢰 높음",
            background=colors["surface_soft"],
            foreground=colors["muted"],
            font=(font, 8),
        ).pack(side="left", padx=(2, 8))
        tk.Label(
            legend,
            text="●",
            background=colors["surface_soft"],
            foreground=colors["warning"],
            font=(font, 8, "bold"),
        ).pack(side="left")
        tk.Label(
            legend,
            text="확인 필요",
            background=colors["surface_soft"],
            foreground=colors["muted"],
            font=(font, 8),
        ).pack(side="left", padx=(2, 0))

        controls_canvas = tk.Canvas(
            controls_shell,
            background=colors["background"],
            highlightthickness=0,
            width=435,
        )
        controls_scrollbar = ttk.Scrollbar(
            controls_shell,
            orient="vertical",
            command=controls_canvas.yview,
            style="Modern.Vertical.TScrollbar",
        )
        controls_canvas.configure(yscrollcommand=controls_scrollbar.set)
        controls_scrollbar.pack(side="right", fill="y")
        controls_canvas.pack(side="left", fill="both", expand=True)
        controls = ttk.Frame(controls_canvas, style="App.TFrame")
        controls_window = controls_canvas.create_window(
            (0, 0), window=controls, anchor="nw"
        )
        controls.bind(
            "<Configure>",
            lambda event: controls_canvas.configure(
                scrollregion=controls_canvas.bbox("all")
            ),
        )
        controls_canvas.bind(
            "<Configure>",
            lambda event: controls_canvas.itemconfigure(
                controls_window, width=event.width
            ),
        )

        source_card = tk.Frame(
            controls,
            background=colors["surface"],
            highlightthickness=0,
            bd=0,
        )
        source_card.pack(fill="x", pady=(0, 10))
        source_header = ttk.Frame(
            source_card, style="Card.TFrame", padding=(14, 12, 14, 7)
        )
        source_header.pack(fill="x")
        tk.Label(
            source_header,
            text="01",
            background=colors["accent_soft"],
            foreground=colors["accent_dark"],
            font=(font, 8, "bold"),
            padx=8,
            pady=3,
        ).pack(side="left", padx=(0, 9))
        ttk.Label(
            source_header, text="화면 연결", style="CardTitle.TLabel"
        ).pack(side="left")
        source_content = ttk.Frame(
            source_card, style="Card.TFrame", padding=(14, 0, 14, 14)
        )
        source_content.pack(fill="x")
        ttk.Label(
            source_content,
            text="iPhone Camo, 웹캠, 사진 또는 데모 화면을 불러옵니다.",
            style="CardHint.TLabel",
            wraplength=395,
        ).pack(anchor="w", pady=(0, 8))

        camera_options = ttk.Frame(source_content, style="Card.TFrame")
        camera_options.pack(fill="x", pady=(0, 8))
        ttk.Label(
            camera_options, text="사용할 카메라", style="Card.TLabel"
        ).pack(anchor="w", pady=(0, 5))
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
            width=24,
            style="Camera.TCombobox",
        )
        self.camera_selector.pack(fill="x")

        input_buttons = ttk.Frame(source_content, style="Card.TFrame")
        input_buttons.pack(fill="x", pady=(1, 0))
        self.camera_button = ttk.Button(
            input_buttons,
            text="카메라 연결",
            command=self.toggle_camera,
            style="Primary.TButton",
        )
        self.camera_button.pack(fill="x")

        alternate_inputs = ttk.Frame(source_content, style="Card.TFrame")
        alternate_inputs.pack(fill="x", pady=(6, 0))
        ttk.Button(
            alternate_inputs,
            text="사진 불러오기",
            command=self.open_image,
            style="Secondary.TButton",
        ).pack(side="left", expand=True, fill="x")
        ttk.Button(
            alternate_inputs,
            text="데모 실행",
            command=self.load_demo,
            style="Secondary.TButton",
        ).pack(side="left", expand=True, fill="x", padx=(6, 0))

        analysis_buttons = ttk.Frame(source_content, style="Card.TFrame")
        analysis_buttons.pack(fill="x", pady=(12, 0))
        ttk.Separator(analysis_buttons, orient="horizontal").pack(
            fill="x", pady=(0, 10)
        )
        self.analyze_button = ttk.Button(
            analysis_buttons,
            text="화면 분석하기 (F5)",
            command=self.analyze_current,
            style="Primary.TButton",
        )
        self.analyze_button.pack(fill="x")
        self.full_frame_button = ttk.Button(
            analysis_buttons,
            text="전체 화면으로 다시 분석",
            command=lambda: self.analyze_current(force_full_frame=True),
            style="Tertiary.TButton",
        )
        self.full_frame_button.pack(anchor="e", pady=(3, 0))

        status_box = tk.Frame(
            source_content,
            background=colors["surface_soft"],
            padx=12,
            pady=10,
        )
        status_box.pack(fill="x", pady=(8, 0))
        camo_available = any(device.is_camo for device in self.camera_devices)
        initial_connection = (
            "● Camo/iPhone 감지됨 · 연결 대기"
            if camo_available
            else "● 카메라 연결 대기"
        )
        self.connection_status_var = tk.StringVar(value=initial_connection)
        self.connection_status_label = tk.Label(
            status_box,
            textvariable=self.connection_status_var,
            wraplength=370,
            foreground="#6B7280",
            background=colors["surface_soft"],
            justify="left",
            anchor="w",
            font=(font, 10, "bold"),
        )
        self.connection_status_label.pack(anchor="w", fill="x")
        self.camera_status_var = tk.StringVar(value="카메라 미사용")
        tk.Label(
            status_box,
            textvariable=self.camera_status_var,
            wraplength=370,
            foreground="#436078",
            background=colors["surface_soft"],
            justify="left",
            anchor="w",
            font=(font, 9),
        ).pack(anchor="w", fill="x", pady=(3, 0))

        self.status_var = tk.StringVar(value="OCR 분석 대기")
        tk.Label(
            status_box,
            textvariable=self.status_var,
            wraplength=370,
            foreground=colors["success"],
            background=colors["surface_soft"],
            justify="left",
            anchor="w",
            font=(font, 9),
        ).pack(anchor="w", fill="x", pady=(2, 0))
        self.analysis_progress = ttk.Progressbar(
            source_content,
            mode="determinate",
            value=0,
            style="Analysis.Horizontal.TProgressbar",
        )
        self.analysis_progress.pack(fill="x", pady=(7, 0))

        target_card = tk.Frame(
            controls,
            background=colors["surface"],
            highlightthickness=0,
            bd=0,
        )
        target_card.pack(fill="x", pady=(0, 10))
        target_header = ttk.Frame(
            target_card, style="Card.TFrame", padding=(14, 12, 14, 7)
        )
        target_header.pack(fill="x")
        tk.Label(
            target_header,
            text="02",
            background=colors["accent_soft"],
            foreground=colors["accent_dark"],
            font=(font, 8, "bold"),
            padx=8,
            pady=3,
        ).pack(side="left", padx=(0, 9))
        ttk.Label(
            target_header, text="메뉴 찾기", style="CardTitle.TLabel"
        ).pack(side="left")
        target_content = ttk.Frame(
            target_card, style="Card.TFrame", padding=(14, 0, 14, 14)
        )
        target_content.pack(fill="x")
        ttk.Label(
            target_content,
            text="찾을 메뉴 이름을 입력하세요. 예: 불고기버거",
            style="CardHint.TLabel",
        ).pack(anchor="w", pady=(0, 7))
        target_input = tk.Frame(target_content, background=colors["surface"])
        target_input.pack(fill="x")
        self.target_query_var = tk.StringVar()
        entry_shell = tk.Frame(
            target_input,
            background="#B8C5D8",
            padx=1,
            pady=1,
        )
        entry_shell.pack(side="left", fill="x", expand=True)
        self.target_entry = tk.Entry(
            entry_shell,
            textvariable=self.target_query_var,
            background="#FFFFFF",
            foreground=colors["text"],
            insertbackground=colors["accent"],
            selectbackground=colors["accent_soft"],
            selectforeground=colors["text"],
            relief="flat",
            borderwidth=0,
            font=(font, 11),
        )
        self.target_entry.pack(fill="both", expand=True, ipady=10, padx=10)
        self.target_entry.bind("<Return>", self.search_target)
        self.target_entry.bind(
            "<FocusIn>", lambda event: entry_shell.configure(background=colors["accent"])
        )
        self.target_entry.bind(
            "<FocusOut>", lambda event: entry_shell.configure(background="#B8C5D8")
        )
        ttk.Button(
            target_input,
            text="메뉴 찾기",
            command=self.search_target,
            style="Primary.TButton",
        ).pack(side="left", padx=(7, 0))

        self.target_status_var = tk.StringVar(
            value="메뉴명을 입력하거나 아래 인식 결과에서 목표를 선택하세요."
        )
        self.guide_var = tk.StringVar(value="분석 후 메뉴를 선택하세요.")
        guide_surface = tk.Frame(
            target_content,
            background=colors["navy"],
            padx=14,
            pady=13,
        )
        guide_surface.pack(fill="x", pady=(10, 0))
        tk.Label(
            guide_surface,
            text="현재 안내",
            font=(font, 8, "bold"),
            foreground="#8EB5FF",
            background=colors["navy"],
            justify="left",
            anchor="w",
        ).pack(anchor="w")

        tk.Label(
            guide_surface,
            textvariable=self.guide_var,
            wraplength=350,
            font=(font, 13, "bold"),
            foreground="#FFFFFF",
            background=colors["navy"],
            justify="left",
            anchor="w",
        ).pack(anchor="w", fill="x", pady=(4, 6))

        tk.Label(
            guide_surface,
            textvariable=self.target_status_var,
            wraplength=350,
            font=(font, 9),
            foreground="#C7D4E7",
            background=colors["navy"],
            justify="left",
            anchor="w",
        ).pack(anchor="w", fill="x", pady=(0, 9))

        self.haptic_panel = tk.Label(
            guide_surface,
            text="가상 진동 안내 · 대기",
            font=(font, 9, "bold"),
            foreground="#D5DFEC",
            background=colors["navy_soft"],
            padx=10,
            pady=8,
            anchor="w",
        )
        self.haptic_panel.pack(fill="x")

        self.tts_var = tk.BooleanVar(value=True)
        speech_controls = ttk.Frame(target_content, style="Card.TFrame")
        speech_controls.pack(fill="x", pady=(7, 0))
        self.tts_toggle = tk.Checkbutton(
            speech_controls,
            text="음성 안내 켜짐",
            variable=self.tts_var,
            command=self._toggle_speech,
            indicatoron=False,
            relief="flat",
            offrelief="flat",
            borderwidth=0,
            background="#F1F5F9",
            activebackground=colors["success_soft"],
            selectcolor=colors["success_soft"],
            foreground=colors["text"],
            activeforeground=colors["success"],
            font=(font, 9, "bold"),
            padx=10,
            pady=7,
            cursor="hand2",
        )
        self.tts_toggle.pack(side="left", anchor="w")
        ttk.Button(
            speech_controls,
            text="안내 듣기 (R)",
            command=self.repeat_current_guidance,
            style="Secondary.TButton",
        ).pack(side="right")

        result_card = tk.Frame(
            controls,
            background=colors["surface"],
            highlightthickness=0,
            bd=0,
        )
        result_card.pack(fill="both", expand=True)
        result_header = ttk.Frame(
            result_card, style="Card.TFrame", padding=(14, 12, 14, 7)
        )
        result_header.pack(fill="x")
        tk.Label(
            result_header,
            text="03",
            background=colors["accent_soft"],
            foreground=colors["accent_dark"],
            font=(font, 8, "bold"),
            padx=8,
            pady=3,
        ).pack(side="left", padx=(0, 9))
        ttk.Label(
            result_header, text="인식 결과", style="CardTitle.TLabel"
        ).pack(side="left")
        self.result_count_var = tk.StringVar(value="분석 대기")
        tk.Label(
            result_header,
            textvariable=self.result_count_var,
            background=colors["accent_soft"],
            foreground=colors["accent_dark"],
            font=(font, 8, "bold"),
            padx=8,
            pady=3,
        ).pack(side="right")
        result_content = ttk.Frame(
            result_card, style="Card.TFrame", padding=(14, 0, 14, 14)
        )
        result_content.pack(fill="both", expand=True)
        ttk.Label(
            result_content,
            text="결과 행을 클릭하거나 방향키와 Enter로 목표를 선택할 수 있습니다.",
            style="CardHint.TLabel",
            wraplength=395,
        ).pack(anchor="w", pady=(0, 7))
        list_frame = tk.Frame(
            result_content,
            background="#FFFFFF",
            highlightthickness=0,
            bd=0,
        )
        list_frame.pack(fill="both", expand=True)
        self.item_list = ttk.Treeview(
            list_frame,
            columns=("menu", "position", "confidence"),
            show="headings",
            selectmode="browse",
            height=6,
            style="Result.Treeview",
            takefocus=True,
        )
        self.item_list.heading("menu", text="메뉴")
        self.item_list.heading("position", text="격자 위치")
        self.item_list.heading("confidence", text="OCR 상태")
        self.item_list.column("menu", width=150, minwidth=105, stretch=True)
        self.item_list.column(
            "position", width=105, minwidth=90, stretch=False, anchor="w"
        )
        self.item_list.column(
            "confidence", width=92, minwidth=80, stretch=False, anchor="center"
        )
        self.item_list.tag_configure("reliable", foreground=colors["text"])
        self.item_list.tag_configure("review", foreground=colors["warning"])
        self.item_list.tag_configure("low", foreground=colors["danger"])
        item_scrollbar = ttk.Scrollbar(
            list_frame,
            orient="vertical",
            command=self.item_list.yview,
            style="Modern.Vertical.TScrollbar",
        )
        self.item_list.configure(yscrollcommand=item_scrollbar.set)
        item_scrollbar.pack(side="right", fill="y")
        self.item_list.pack(side="left", fill="both", expand=True)
        self.result_empty_label = tk.Label(
            list_frame,
            text="화면을 분석하면 인식된 메뉴가\n여기에 정리됩니다.",
            background="#FFFFFF",
            foreground="#94A3B8",
            font=(font, 10),
            justify="center",
        )
        self.result_empty_label.place(relx=0.5, rely=0.58, anchor="center")
        self.item_list.bind("<<TreeviewSelect>>", self.select_item)
        self.item_list.bind("<Return>", self.select_item)

        def scroll_control_panel(event):
            if event.widget is self.item_list:
                return None
            pointer_x = self.root.winfo_pointerx()
            panel_left = controls_canvas.winfo_rootx()
            panel_right = panel_left + controls_canvas.winfo_width()
            if not (panel_left <= pointer_x <= panel_right):
                return None
            if controls.winfo_reqheight() <= controls_canvas.winfo_height():
                return None
            direction = -3 if event.delta > 0 else 3
            controls_canvas.yview_scroll(direction, "units")
            return "break"

        def is_control_widget(widget) -> bool:
            current = widget
            while current is not None:
                if current is controls:
                    return True
                current = getattr(current, "master", None)
            return False

        def reveal_focused_control(widget) -> None:
            if not widget.winfo_exists() or not is_control_widget(widget):
                return
            controls_canvas.update_idletasks()
            viewport_top = controls_canvas.canvasy(0)
            viewport_height = controls_canvas.winfo_height()
            widget_top = widget.winfo_rooty() - controls.winfo_rooty()
            widget_bottom = widget_top + widget.winfo_height()
            content_height = max(controls.winfo_reqheight(), 1)
            if widget_top < viewport_top:
                controls_canvas.yview_moveto(max(0.0, widget_top / content_height))
            elif widget_bottom > viewport_top + viewport_height:
                target_top = widget_bottom - viewport_height
                controls_canvas.yview_moveto(
                    min(1.0, max(0.0, target_top / content_height))
                )

        def keep_keyboard_focus_visible(event):
            if is_control_widget(event.widget):
                self.root.after_idle(reveal_focused_control, event.widget)

        self.root.bind_all("<MouseWheel>", scroll_control_panel, add="+")
        self.root.bind_all("<FocusIn>", keep_keyboard_focus_visible, add="+")
        ttk.Label(
            result_content,
            text="신뢰도가 낮은 항목은 결과 표에서 직접 확인한 뒤 선택하세요.",
            style="CardHint.TLabel",
            wraplength=395,
        ).pack(anchor="w", pady=(7, 0))

        footer = tk.Frame(
            self.root,
            background=colors["background"],
            padx=20,
            pady=8,
        )
        footer.pack(fill="x")
        tk.Label(
            footer,
            text="단축키",
            background=colors["background"],
            foreground=colors["text"],
            font=(font, 9, "bold"),
        ).pack(side="left")
        tk.Label(
            footer,
            text="F2 카메라  ·  F3 사진  ·  F4 데모  ·  F5 분석  ·  R 안내  ·  Esc 중지",
            background=colors["background"],
            foreground=colors["muted"],
            font=(font, 9),
        ).pack(side="left", padx=(8, 0))
        tk.Label(
            footer,
            text="가상 진동 시뮬레이션",
            background=colors["warning_soft"],
            foreground=colors["warning"],
            font=(font, 9, "bold"),
            padx=10,
            pady=4,
        ).pack(side="right")

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
        enabled = self.tts_var.get()
        self.speaker.set_enabled(enabled)
        if hasattr(self, "tts_toggle"):
            self.tts_toggle.configure(
                text="음성 안내 켜짐" if enabled else "음성 안내 꺼짐"
            )

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
        self._clear_result_list()
        self.result_count_var.set("분석 준비")
        self.target_query_var.set("")
        self._reset_tracking_feedback()
        self._set_connection_status(
            self._idle_connection_message(), "idle"
        )
        self.camera_status_var.set("내장 데모 사용 · 카메라 미사용")
        self.status_var.set("내장 모의 키오스크를 분석합니다.")
        self.analyze_current(auto=True)

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
        self._clear_result_list()
        self.result_count_var.set("분석 대기")
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
        self._clear_result_list()
        self.result_count_var.set("연결 대기")
        self._reset_tracking_feedback()
        self.screen_tracker.reset()
        self.stability.reset()
        self.screen_miss_count = 0
        self._clear_frame_candidates()
        self.camera_generation += 1
        generation = self.camera_generation
        self.camera_button.configure(text="연결 취소")
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
            self.camera_button.configure(text="카메라 연결")
            self.result_count_var.set("연결 실패")
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
        self.camera_button.configure(text="카메라 중지")
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
            self.camera_button.configure(text="카메라 연결")
        if hasattr(self, "camera_selector"):
            self.camera_selector.configure(state="readonly")
        if announce and was_active:
            disconnected = (
                "Camo/iPhone 연결 해제"
                if was_camo
                else "카메라 연결 해제"
            )
            self._set_connection_status(disconnected, "idle")
            self.camera_status_var.set("카메라를 중지했습니다.")
            self._reset_tracking_feedback()
            self._speak_throttled("카메라를 중지했습니다.", force=True)
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
                    "네 모서리가 보이도록 다시 비추거나 "
                    "'전체 화면으로 다시 분석'을 누르세요."
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
        self.analysis_progress.configure(mode="indeterminate")
        self.analysis_progress.start(12)
        self.result_count_var.set("분석 중")
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
        self.analysis_progress.stop()
        self.analysis_progress.configure(mode="determinate", value=0)
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
                    " 화면 테두리가 보이지 않으면 "
                    "'전체 화면으로 다시 분석'을 사용하세요."
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

    def _release_result_selection_guard(self) -> None:
        self._result_selection_guard = False

    def _clear_result_list(self) -> None:
        if not hasattr(self, "item_list"):
            return
        self._result_selection_guard = True
        rows = self.item_list.get_children()
        if rows:
            self.item_list.delete(*rows)
        if hasattr(self, "result_empty_label"):
            self.result_empty_label.place(relx=0.5, rely=0.58, anchor="center")
            self.result_empty_label.lift()
        self.root.after_idle(self._release_result_selection_guard)

    def _set_result_selection(self, index: int) -> None:
        row_id = str(index)
        if not self.item_list.exists(row_id):
            return
        self._result_selection_guard = True
        self.item_list.selection_set(row_id)
        self.item_list.focus(row_id)
        self.item_list.see(row_id)
        self.root.after_idle(self._release_result_selection_guard)

    def _selected_result_index(self) -> int | None:
        selected = self.item_list.selection()
        if not selected:
            return None
        try:
            return int(selected[0])
        except (TypeError, ValueError):
            return None

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
        self._clear_result_list()
        for index, item in enumerate(result.items):
            confidence_label = self._confidence_label(item.confidence)
            self.item_list.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    item.text,
                    f"{item.grid} {item.relative}".strip(),
                    f"{confidence_label}  {item.confidence:.0f}%",
                ),
                tags=(
                    "reliable"
                    if item.confidence >= 70
                    else "review"
                    if item.confidence >= 50
                    else "low",
                ),
            )
        if result.items:
            self.result_empty_label.place_forget()
        mode = "전체 프레임" if request["force_full_frame"] else "화면 검출"
        reliable_count = sum(item.confidence >= 70 for item in result.items)
        review_count = len(result.items) - reliable_count
        self.result_count_var.set(f"{len(result.items)}개")
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
                        f"OCR {candidate.confidence:.0f}%. 결과 행을 클릭해 확인하세요."
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
                    "메뉴명을 입력하거나 아래 인식 결과에서 목표를 선택하세요."
                )
            self.guide_var.set(
                "인식된 메뉴가 없습니다."
                if not result.items
                else "찾을 메뉴를 입력하거나 결과 표에서 목표를 선택하세요."
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
        self._set_result_selection(index)

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
            "결과 행을 클릭하거나 엔터를 눌러 확인하세요."
        )
        self.target_status_var.set(message)
        self.guide_var.set(
            "후보를 확인하면 목표 추적과 가상 진동 안내가 시작됩니다."
        )
        self._render_current()
        if speak:
            self._speak_throttled(
                f"후보는 {candidate.text}입니다. 결과 표에서 확인하세요.",
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
        if self._result_selection_guard or not self.result or not self.result.items:
            return
        index = self._selected_result_index()
        if index is None:
            return
        self._activate_item(index, speak=True)

    def _activate_item(
        self,
        index: int,
        speak: bool,
        preserve_pointer: bool = False,
    ) -> None:
        if not self.result or not (0 <= index < len(self.result.items)):
            return
        preserved_pointer = self.manual_pointer if preserve_pointer else None
        self._set_result_selection(index)
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
                f"{self.selected_item.text}, 화면 {position}에 있습니다."
            )
            detail_message = (
                f"촉각 격자 {self.selected_item.grid}{relative} · "
                f"좌표 ({grid_x}, {grid_y}) · "
                f"OCR 신뢰도 {self.selected_item.confidence:.0f}%"
            )
            voice_message = (
                f"{self.selected_item.text}, 화면 {position}에 있습니다."
            )
            self.location_voice_message = voice_message
            self.guide_var.set(display_message)
            if self.capture is not None:
                self.target_status_var.set(
                    f"{detail_message} · "
                    "화면을 클릭하면 그 위치를 기준으로 다시 안내합니다."
                )
            else:
                self.target_status_var.set(
                    f"{detail_message} · "
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
            detail = "목표 안쪽 · 가상 연속 진동 표시"
        else:
            detail = (
                f"목표 거리 {normalized:.1%} · "
                f"가상 진동 간격 약 {interval} ms"
            )
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
            "클릭 위치가 목표 안쪽 · 가상 연속 진동 표시"
            if inside
            else (
                f"클릭 위치 기준 거리 {normalized:.1%} · "
                f"가상 진동 간격 약 {interval} ms"
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
        self.haptic_detail = "목표를 선택하면 가상 진동 안내가 시작됩니다."
        if hasattr(self, "haptic_var"):
            self.haptic_var.set(self.haptic_detail)
        if not keep_target and hasattr(self, "target_status_var"):
            self.manual_pointer = None
            self.location_voice_message = ""
            self.target_status_var.set(
                "메뉴명을 입력하거나 아래 인식 결과에서 목표를 선택하세요."
            )

    def _haptic_tick(self) -> None:
        if self.closed or not self.haptic_panel.winfo_exists():
            return
        colors = self.ui_colors
        if not self.haptic_active:
            self.haptic_panel.configure(
                text="가상 진동 안내 · 대기",
                background=colors["navy_soft"],
                foreground="#D5DFEC",
            )
        elif self.haptic_inside:
            self.haptic_panel.configure(
                text="목표 도달 · 가상 연속 진동 표시",
                background=colors["success"],
                foreground="#FFFFFF",
            )
        else:
            if self.haptic_interval <= 180:
                stage = "매우 가까움"
                background = colors["warning_soft"]
                foreground = colors["warning"]
            elif self.haptic_interval <= 420:
                stage = "접근 중"
                background = colors["accent_soft"]
                foreground = colors["accent_dark"]
            else:
                stage = "목표까지 이동 중"
                background = colors["navy_soft"]
                foreground = "#D5DFEC"
            self.haptic_panel.configure(
                text=(
                    f"가상 진동 시뮬레이션 · {stage} · "
                    f"약 {self.haptic_interval} ms"
                ),
                background=background,
                foreground=foreground,
            )
        self.root.after(120, self._haptic_tick)

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
