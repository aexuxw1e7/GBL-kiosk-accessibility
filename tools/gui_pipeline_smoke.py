from __future__ import annotations

import sys
import time
import tkinter as tk
import traceback
from copy import copy
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
            app.start_mission()
            mission_started = app.mission is not None and app.mission.running
            original_item = app.selected_item
            original_target_rect = app.mission.target_rect
            duplicate = copy(original_item)
            duplicate_index = len(app.result.items)
            app.result.items.append(duplicate)
            app.item_list.insert(
                "",
                "end",
                iid=str(duplicate_index),
                values=(duplicate.text, "중복 메뉴", "테스트"),
            )
            app._set_result_selection(duplicate_index)
            root.update_idletasks()
            app.select_item()
            duplicate_blocked = (
                app.selected_item is original_item
                and app.mission is not None
                and app.mission.target_rect == original_target_rect
            )
            app._register_mission_click((0.0, 0.0))
            app._observe_mission_point(
                app.selected_item.center,
                "cursor",
                complete_on_inside=True,
                force_sample=True,
            )
            mission_result = (
                app.mission.snapshot(time.monotonic())
                if app.mission is not None
                else None
            )
            mission_ok = (
                mission_started
                and app.mission is not None
                and app.mission.succeeded
                and mission_result is not None
                and mission_result.wrong_clicks == 1
                and len(mission_result.route_segments) == 1
                and len(app.mission_history) == 1
                and app.mission_phase_var.get() == "미션 성공"
            )
            app.start_mission()

            class ReleasableCapture:
                released = False

                def release(self):
                    self.released = True

            fake_capture = ReleasableCapture()
            app.capture = fake_capture
            app.stop_camera()
            camera_stop_reset = (
                fake_capture.released
                and app.mission is None
                and "미션을 초기화" in app.mission_result_var.get()
            )
            state["ok"] = (
                selected_ok
                and coordinate_once
                and r_once
                and click_ok
                and click_silent
                and pointer_preserved
                and click_r_once
                and mission_ok
                and duplicate_blocked
                and camera_stop_reset
            )
            state["message"] = (
                f"{len(app.result.items)} items / target={target!r} / "
                f"selected={getattr(app.selected_item, 'text', None)!r} / "
                f"spoken={spoken!r} / click={app.location_voice_message!r} / "
                f"duplicate_blocked={duplicate_blocked!r} / "
                f"camera_stop_reset={camera_stop_reset!r}"
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
