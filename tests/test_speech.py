import unittest
from unittest.mock import patch

from kiosk_accessibility.speech import (
    DEFAULT_NEURAL_VOICE,
    POWERSHELL_SPEECH_SCRIPT,
    Speaker,
)


class FakeProcess:
    def __init__(self):
        self.returncode = None

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def kill(self):
        self.returncode = -9


class SpeechTests(unittest.TestCase):
    def test_text_is_passed_by_environment_not_powershell_source(self):
        self.assertIn("$env:GBL_TTS_TEXT", POWERSHELL_SPEECH_SCRIPT)
        self.assertNotIn("$args[0]", POWERSHELL_SPEECH_SCRIPT)

        with patch(
            "kiosk_accessibility.speech.subprocess.Popen",
            return_value=FakeProcess(),
        ) as popen:
            speaker = Speaker(enabled=True, prefer_neural=False)
            speaker.speak("불고기버거, 화면 왼쪽 위에 있습니다.", interrupt=True)
            self.assertTrue(speaker.wait_until_idle(timeout=2))
            speaker.stop()

        arguments, keywords = popen.call_args
        self.assertNotIn("불고기버거", " ".join(arguments[0]))
        self.assertEqual(
            keywords["env"]["GBL_TTS_TEXT"],
            "불고기버거, 화면 왼쪽 위에 있습니다.",
        )
        self.assertEqual(speaker.last_exit_code, 0)

    def test_neural_voice_is_used_without_local_fallback(self):
        with (
            patch.object(Speaker, "_speak_neural") as neural,
            patch.object(Speaker, "_speak_windows") as local,
        ):
            speaker = Speaker(enabled=True, prefer_neural=True)
            speaker.prefer_neural = True
            speaker.speak("새우버거, 오른쪽에 있습니다.", interrupt=True)
            self.assertTrue(speaker.wait_until_idle(timeout=2))
            speaker.stop()

        neural.assert_called_once()
        local.assert_not_called()
        self.assertEqual(speaker.last_backend, "edge-neural")
        self.assertEqual(speaker.neural_voice, DEFAULT_NEURAL_VOICE)

    def test_neural_failure_uses_local_fallback(self):
        with (
            patch.object(
                Speaker,
                "_speak_neural",
                side_effect=RuntimeError("network unavailable"),
            ) as neural,
            patch.object(Speaker, "_speak_windows") as local,
        ):
            speaker = Speaker(enabled=True, prefer_neural=True)
            speaker.prefer_neural = True
            speaker.speak("치킨버거, 아래에 있습니다.", interrupt=True)
            self.assertTrue(speaker.wait_until_idle(timeout=2))
            speaker.stop()

        neural.assert_called_once()
        local.assert_called_once()
        self.assertIn("network unavailable", speaker.last_error)


if __name__ == "__main__":
    unittest.main()
