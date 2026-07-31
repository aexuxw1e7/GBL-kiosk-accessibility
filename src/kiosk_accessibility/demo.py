from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .models import OCRItem


SCREEN_SIZE = (600, 1000)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path(r"C:\Windows\Fonts\malgunbd.ttf" if bold else r"C:\Windows\Fonts\malgun.ttf"),
        Path(r"C:\Windows\Fonts\NanumGothicBold.ttf" if bold else r"C:\Windows\Fonts\NanumGothic.ttf"),
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return ImageFont.load_default()
    return ImageFont.truetype(str(path), size)


def create_mock_kiosk() -> tuple[np.ndarray, list[OCRItem]]:
    width, height = SCREEN_SIZE
    screen = Image.new("RGB", SCREEN_SIZE, "#F7F9FC")
    draw = ImageDraw.Draw(screen)
    title_font = _font(34, bold=True)
    menu_font = _font(25, bold=True)
    price_font = _font(20)
    small_font = _font(18)

    draw.rectangle((0, 0, width, 118), fill="#173B57")
    draw.text((36, 30), "오늘의 버거 메뉴", font=title_font, fill="white")
    draw.text((36, 78), "원하는 메뉴를 선택하세요", font=small_font, fill="#D6E8F5")

    menu_data = [
        ("불고기버거", "6,500원"),
        ("새우버거", "7,000원"),
        ("치킨버거", "7,500원"),
        ("치즈버거", "6,800원"),
        ("채식버거", "8,000원"),
        ("감자튀김", "3,000원"),
    ]
    expected: list[OCRItem] = []
    card_width, card_height = 250, 205
    x_positions = [32, 318]
    y_positions = [150, 380, 610]
    colors = ["#EAF5FF", "#FFF2DF", "#EAF8EF", "#F5EDFF", "#FFF0F3", "#FFF8D8"]
    for index, ((name, price), color) in enumerate(zip(menu_data, colors)):
        column, row = index % 2, index // 2
        x, y = x_positions[column], y_positions[row]
        draw.rounded_rectangle(
            (x, y, x + card_width, y + card_height),
            radius=20,
            fill=color,
            outline="#B4C4D2",
            width=3,
        )
        draw.ellipse((x + 82, y + 20, x + 168, y + 106), fill="#D8A45A")
        draw.rectangle((x + 88, y + 55, x + 162, y + 68), fill="#5F9D52")
        text_bbox = draw.textbbox((0, 0), name, font=menu_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = x + (card_width - text_width) // 2
        text_y = y + 120
        draw.text((text_x, text_y), name, font=menu_font, fill="#14212B")
        price_bbox = draw.textbbox((0, 0), price, font=price_font)
        price_width = price_bbox[2] - price_bbox[0]
        draw.text(
            (x + (card_width - price_width) // 2, y + 160),
            price,
            font=price_font,
            fill="#465B6B",
        )
        expected.append(
            OCRItem(
                text=name,
                x=text_x,
                y=text_y,
                width=text_width,
                height=text_bbox[3] - text_bbox[1] + 8,
                confidence=99.0,
            )
        )

    draw.rounded_rectangle((32, 865, 282, 955), radius=18, fill="#E6EBEF")
    draw.rounded_rectangle((318, 865, 568, 955), radius=18, fill="#147D64")
    draw.text((112, 890), "이전", font=menu_font, fill="#23343F")
    draw.text((388, 890), "결제하기", font=menu_font, fill="white")
    expected.extend(
        [
            OCRItem("이전", 112, 890, 58, 35, 99.0),
            OCRItem("결제하기", 388, 890, 100, 35, 99.0),
        ]
    )

    screen_bgr = cv2.cvtColor(np.asarray(screen), cv2.COLOR_RGB2BGR)
    frame_height, frame_width = 1200, 900
    frame = np.full((frame_height, frame_width, 3), (31, 37, 42), dtype=np.uint8)
    source = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    destination = np.array(
        [[155, 76], [742, 126], [792, 1094], [104, 1136]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(screen_bgr, matrix, (frame_width, frame_height))
    mask = cv2.warpPerspective(
        np.full((height, width), 255, dtype=np.uint8),
        matrix,
        (frame_width, frame_height),
    )
    frame[mask > 0] = warped[mask > 0]
    cv2.polylines(
        frame,
        [destination.astype(np.int32).reshape(-1, 1, 2)],
        True,
        (92, 101, 108),
        18,
        cv2.LINE_AA,
    )
    return frame, expected

