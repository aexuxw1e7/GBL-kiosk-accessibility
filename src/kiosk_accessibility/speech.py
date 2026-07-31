from __future__ import annotations

import hashlib
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import edge_tts as _edge_tts
except ImportError:
    _edge_tts = None

try:
    import pygame as _pygame
except ImportError:
    _pygame = None


POWERSHELL_SPEECH_SCRIPT = (
    "Add-Type -AssemblyName System.Speech; "
    "$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "try { "
    "$korean = $voice.GetInstalledVoices() | "
    "Where-Object { $_.Enabled -and $_.VoiceInfo.Culture.Name -eq 'ko-KR' } | "
    "Select-Object -First 1; "
    "if ($null -ne $korean) { $voice.SelectVoice($korean.VoiceInfo.Name) }; "
    "$voice.Rate = 0; $voice.Volume = 100; "
    "$voice.SetOutputToDefaultAudioDevice(); "
    "$voice.Speak($env:GBL_TTS_TEXT) "
    "} finally { $voice.Dispose() }"
)

EDGE_TTS_SYNTHESIS_SCRIPT = (
    "import asyncio, os; "
    "from edge_tts import Communicate; "
    "asyncio.run(Communicate("
    "os.environ['GBL_TTS_TEXT'], "
    "voice=os.environ['GBL_TTS_VOICE'], "
    "rate=os.environ['GBL_TTS_RATE']"
    ").save(os.environ['GBL_TTS_OUTPUT']))"
)

DEFAULT_NEURAL_VOICE = "ko-KR-SunHiNeural"


class _SpeechCancelled(Exception):
    pass


class Speaker:
    """Play requested Korean guidance with neural TTS and a local fallback."""

    def __init__(
        self,
        enabled: bool = True,
        prefer_neural: bool = True,
        neural_voice: str = DEFAULT_NEURAL_VOICE,
        neural_timeout: float = 12.0,
    ) -> None:
        self.enabled = enabled
        self.prefer_neural = (
            prefer_neural and _edge_tts is not None and _pygame is not None
        )
        self.neural_voice = neural_voice
        self.neural_timeout = neural_timeout
        self.process: subprocess.Popen | None = None
        self.last_exit_code: int | None = None
        self.last_backend: str | None = None
        self.last_error: str | None = None
        self._messages: queue.Queue[tuple[str, int] | None] = queue.Queue(
            maxsize=1
        )
        self._lock = threading.Lock()
        self._generation = 0
        self._closed = False
        self._busy = threading.Event()
        self._playback_cancel = threading.Event()
        self._mixer_initialized = False
        self._worker = threading.Thread(
            target=self._run,
            name="gbl-korean-tts",
            daemon=True,
        )
        self._worker.start()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        if not enabled:
            self.cancel()

    def speak(self, text: str, interrupt: bool = False) -> None:
        if (
            not self.enabled
            or not text
            or sys.platform != "win32"
            or self._closed
        ):
            return
        if interrupt:
            self.cancel()
        else:
            self._discard_pending()
        with self._lock:
            generation = self._generation
        try:
            self._messages.put_nowait((text, generation))
        except queue.Full:
            self._discard_pending()
            try:
                self._messages.put_nowait((text, generation))
            except queue.Full:
                pass

    def _discard_pending(self) -> None:
        while True:
            try:
                self._messages.get_nowait()
                self._messages.task_done()
            except queue.Empty:
                return

    @staticmethod
    def _terminate_specific_process(process: subprocess.Popen | None) -> None:
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass

    def _terminate_process(self) -> None:
        with self._lock:
            process = self.process
        self._terminate_specific_process(process)

    def cancel(self) -> None:
        with self._lock:
            self._generation += 1
        self._playback_cancel.set()
        self._discard_pending()
        self._terminate_process()

    def _is_stale(self, generation: int) -> bool:
        with self._lock:
            return self._closed or generation != self._generation

    def _set_process(self, process: subprocess.Popen | None) -> None:
        with self._lock:
            self.process = process

    @staticmethod
    def _creation_flags() -> int:
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)

    def _neural_cache_path(self, text: str) -> Path:
        cache_dir = Path(tempfile.gettempdir()) / "gbl-kiosk-neural-tts"
        cache_dir.mkdir(parents=True, exist_ok=True)
        identity = f"{self.neural_voice}\0-6%\0{text}".encode("utf-8")
        digest = hashlib.sha256(identity).hexdigest()
        return cache_dir / f"{digest}.mp3"

    def _synthesize_neural(
        self, text: str, destination: Path, generation: int
    ) -> None:
        if _edge_tts is None or self._is_stale(generation):
            raise _SpeechCancelled

        child_environment = os.environ.copy()
        child_environment["GBL_TTS_TEXT"] = text
        child_environment["GBL_TTS_VOICE"] = self.neural_voice
        child_environment["GBL_TTS_RATE"] = "-6%"
        child_environment["GBL_TTS_OUTPUT"] = str(destination)
        module_root = str(Path(_edge_tts.__file__).resolve().parents[1])
        existing_python_path = child_environment.get("PYTHONPATH")
        child_environment["PYTHONPATH"] = (
            module_root
            if not existing_python_path
            else module_root + os.pathsep + existing_python_path
        )

        process = subprocess.Popen(
            [sys.executable, "-c", EDGE_TTS_SYNTHESIS_SCRIPT],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=self._creation_flags(),
            env=child_environment,
        )
        self._set_process(process)
        try:
            try:
                exit_code = process.wait(timeout=self.neural_timeout)
            except subprocess.TimeoutExpired as error:
                self._terminate_specific_process(process)
                raise RuntimeError(
                    "신경망 음성 합성 시간이 초과되었습니다."
                ) from error
        finally:
            with self._lock:
                if self.process is process:
                    self.process = None

        if self._is_stale(generation):
            raise _SpeechCancelled
        if exit_code != 0:
            raise RuntimeError(
                f"신경망 음성 합성 프로세스가 {exit_code} 코드로 종료되었습니다."
            )
        if not destination.exists() or destination.stat().st_size == 0:
            raise RuntimeError("신경망 음성 파일이 생성되지 않았습니다.")

    def _play_neural_file(self, path: Path, generation: int) -> None:
        if _pygame is None or self._is_stale(generation):
            raise _SpeechCancelled
        if not self._mixer_initialized:
            _pygame.mixer.init()
            self._mixer_initialized = True
        _pygame.mixer.music.load(str(path))
        try:
            _pygame.mixer.music.play()
            while _pygame.mixer.music.get_busy():
                if (
                    self._playback_cancel.wait(0.04)
                    or self._is_stale(generation)
                ):
                    _pygame.mixer.music.stop()
                    raise _SpeechCancelled
        finally:
            try:
                _pygame.mixer.music.unload()
            except (AttributeError, _pygame.error):
                pass

    def _speak_neural(self, text: str, generation: int) -> None:
        cache_path = self._neural_cache_path(text)
        partial_path: Path | None = None
        try:
            if not cache_path.exists() or cache_path.stat().st_size == 0:
                handle = tempfile.NamedTemporaryFile(
                    prefix="gbl-tts-",
                    suffix=".mp3",
                    dir=cache_path.parent,
                    delete=False,
                )
                partial_path = Path(handle.name)
                handle.close()
                self._synthesize_neural(text, partial_path, generation)
                if self._is_stale(generation):
                    raise _SpeechCancelled
                partial_path.replace(cache_path)
                partial_path = None
            self._play_neural_file(cache_path, generation)
        finally:
            if partial_path is not None:
                try:
                    partial_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _speak_windows(self, text: str, generation: int) -> None:
        if self._is_stale(generation):
            raise _SpeechCancelled
        child_environment = os.environ.copy()
        child_environment["GBL_TTS_TEXT"] = text
        process = subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                POWERSHELL_SPEECH_SCRIPT,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=self._creation_flags(),
            env=child_environment,
        )
        self._set_process(process)
        self.last_backend = "windows-local"
        try:
            self.last_exit_code = process.wait()
        finally:
            with self._lock:
                if self.process is process:
                    self.process = None
        if self._is_stale(generation):
            raise _SpeechCancelled

    def _speak_message(self, text: str, generation: int) -> None:
        self.last_error = None
        if self.prefer_neural:
            try:
                self._speak_neural(text, generation)
            except _SpeechCancelled:
                raise
            except Exception as error:
                self.last_error = str(error)
            else:
                self.last_backend = "edge-neural"
                self.last_exit_code = 0
                return
        if self._is_stale(generation):
            raise _SpeechCancelled
        self._speak_windows(text, generation)

    def _shutdown_mixer(self) -> None:
        if _pygame is None or not self._mixer_initialized:
            return
        try:
            _pygame.mixer.music.stop()
            _pygame.mixer.quit()
        except _pygame.error:
            pass
        self._mixer_initialized = False

    def _run(self) -> None:
        while True:
            request = self._messages.get()
            if request is None:
                self._messages.task_done()
                self._shutdown_mixer()
                return
            text, generation = request
            if self._is_stale(generation):
                self._messages.task_done()
                continue
            self._busy.set()
            self._playback_cancel.clear()
            try:
                self._speak_message(text, generation)
            except _SpeechCancelled:
                pass
            except Exception as error:
                self.last_exit_code = -1
                self.last_error = str(error)
            finally:
                self._busy.clear()
                self._messages.task_done()

    def wait_until_idle(self, timeout: float = 15.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._busy.is_set() and self._messages.unfinished_tasks == 0:
                return True
            time.sleep(0.05)
        return False

    def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.cancel()
        try:
            self._messages.put_nowait(None)
        except queue.Full:
            self._discard_pending()
            try:
                self._messages.put_nowait(None)
            except queue.Full:
                pass
        if (
            self._worker.is_alive()
            and threading.current_thread() is not self._worker
        ):
            self._worker.join(timeout=2.0)
