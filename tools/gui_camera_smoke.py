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
    gui.Speaker.speak = lambda self, text, interrupt=False: None
    root = tk.Tk()
    root.withdraw()
    app = gui.PrototypeApp(root, ROOT)
    deadline = time.time() + 20
    starting_frame_id = app.current_frame_id
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
                app.close()
                return
            root.after(100, check)
            return
        if (
            app.capture is not None
            and app.current_frame is not None
            and app.current_frame_id >= starting_frame_id + 2
            and not gui.frame_is_blank(app.current_frame)
        ):
            state["ok"] = True
            state["message"] = app.camera_description
            state["settle_until"] = time.time() + 5
            root.after(100, check)
            return
        if time.time() >= deadline:
            state["message"] = app.camera_status_var.get()
            app.close()
            return
        root.after(100, check)

    root.after(100, app.toggle_camera)
    root.after(200, check)
    root.mainloop()
    print(f"GUI_CAMERA_OK={state['ok']} {state['message']}")
    raise SystemExit(0 if state["ok"] else 1)


if __name__ == "__main__":
    main()
