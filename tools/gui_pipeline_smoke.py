from __future__ import annotations

import sys
import time
import tkinter as tk
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
versioned_vendor = ROOT / f".vendor-py{sys.version_info.major}{sys.version_info.minor}"
for candidate in (versioned_vendor, ROOT / ".vendor"):
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break
sys.path.insert(0, str(ROOT / "src"))

import kiosk_accessibility.gui as gui  # noqa: E402


def main() -> None:
    spoken = []

    def record_speech(self, text, interrupt=False):
        spoken.append((text, interrupt))

    gui.Speaker.speak = record_speech
    gui.TesseractOCR.available = property(lambda self: False)
    root = tk.Tk()
    root.withdraw()
    app = gui.PrototypeApp(root, ROOT)
    deadline = time.time() + 25
    state = {"ok": False, "message": "", "settle_until": None, "errors": []}

    def report_callback_exception(exc_type, value, tb) -> None:
        state["errors"].append(
            "".join(traceback.format_exception(exc_type, value, tb))
        )

    root.report_callback_exception = report_callback_exception

    def check() -> None:
        if state["errors"]:
            state["message"] = state["errors"][0]
            app.close()
            return
        if state["settle_until"] is not None:
            if time.time() >= state["settle_until"]:
                state["ok"] = state["ok"] and len(spoken) == 3
                app.close()
                return
            root.after(100, check)
            return
        if (
            not app.analysis_in_progress
            and app.result is not None
            and app.result.items
        ):
            target = app.result.items[-1].text
            app.target_query_var.set(target)
            app.search_target()
            selected_ok = (
                app.selected_item is not None and app.selected_item.text == target
            )
            grid_x, grid_y = gui.grid_numeric_coordinate(app.selected_item)
            coordinate_message = (
                f"현재 {target}는 좌표 {grid_x}, {grid_y}에 있습니다."
            )
            coordinate_once = (
                len(spoken) == 1
                and spoken[0] == (coordinate_message, True)
            )
            initial_message = app.location_voice_message
            r_result = app._global_keypress(
                type("Key", (), {"keysym": "Hangul", "keycode": 82})()
            )
            r_once = (
                len(spoken) == 2
                and spoken[1][0] == initial_message
                and spoken[1][1] is True
                and r_result == "break"
                and bool(root.bind_all("<KeyPress>"))
            )
            app.view_mode.set("corrected")
            app.display_geometry = (0, 0, 600, 1000, 600, 1000)
            click = type("Click", (), {"x": 10, "y": 10})()
            app._canvas_click(click)
            click_message = app.location_voice_message
            click_ok = click_message.endswith(
                "오른쪽 아래에 있습니다."
            )
            click_silent = len(spoken) == 2
            pointer_before = app.manual_pointer
            selected_index = next(
                index
                for index, item in enumerate(app.result.items)
                if item is app.selected_item
            )
            app._activate_item(
                selected_index,
                speak=False,
                preserve_pointer=True,
            )
            pointer_preserved = (
                app.manual_pointer == pointer_before
                and app.location_voice_message == click_message
            )
            click_r_result = app._global_keypress(
                type("Key", (), {"keysym": "r", "keycode": 82})()
            )
            click_r_once = (
                len(spoken) == 3
                and spoken[-1][0] == click_message
                and spoken[-1][1] is True
                and click_r_result == "break"
            )
            state["ok"] = (
                selected_ok
                and coordinate_once
                and r_once
                and click_ok
                and click_silent
                and pointer_preserved
                and click_r_once
            )
            state["message"] = (
                f"{len(app.result.items)} items / target={target!r} / "
                f"selected={getattr(app.selected_item, 'text', None)!r} / "
                f"spoken={spoken!r} / click={app.location_voice_message!r}"
            )
            state["settle_until"] = time.time() + 2
            root.after(100, check)
            return
        if time.time() >= deadline:
            state["message"] = app.status_var.get()
            app.close()
            return
        root.after(100, check)

    root.after(100, check)
    root.mainloop()
    print(f"GUI_PIPELINE_OK={state['ok']} {state['message']}")
    raise SystemExit(0 if state["ok"] else 1)


if __name__ == "__main__":
    main()
