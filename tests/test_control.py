import threading
import time
import unittest

from src.control import AutoControlStateMachine
from src.hardware.command_queue import SerialCommandQueue


CONTINUOUS = {
    "search_speed": 10,
    "color_speed": 5,
    "black_speed": 8,
    "black_threshold": 0.5,
}
STEP = {
    "first_ms": 100,
    "cycle_ms": 50,
    "pause_ms": 20,
    "speed": 5,
    "black_threshold": 0.5,
}
SAFETY = {"max_run_seconds": 10, "black_confirm_frames": 2, "max_missing_frames": 3}


class AutoControlStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.machine = AutoControlStateMachine()

    def update(self, now, color=0.0, black=0.0, connected=True,
               params=CONTINUOUS, safety=SAFETY):
        return self.machine.update(
            color_conf=color, black_conf=black, connected=connected,
            params=params, safety=safety, now=now)

    def test_continuous_mode_only_changes_speed_on_transitions(self):
        self.machine.start("continuous", 0.0)
        first = self.update(0.0, color=0.1)
        self.assertEqual(first.commands, (("set_speed", 10), ("start", None)))
        color = self.update(0.1, color=0.8)
        self.assertEqual(color.commands, (("set_speed", 5),))
        self.assertEqual(self.update(0.2, color=0.8).commands, ())

    def test_continuous_mode_locks_after_confirmed_black(self):
        self.machine.start("continuous", 0.0)
        self.update(0.0, color=0.8)
        self.update(0.1, color=0.8)
        self.assertEqual(self.update(0.2, color=0.8, black=0.8).commands, ())
        enter_black = self.update(0.3, color=0.8, black=0.8)
        self.assertEqual(enter_black.commands, (("set_speed", 8),))
        self.assertEqual(self.update(0.4, black=0.8).commands, ())
        locked = self.update(0.5, black=0.8)
        self.assertEqual(locked.commands, (("stop", None),))
        self.assertEqual(locked.status, "自动控制: 已锁定")

    def test_missing_frames_and_disconnect_stop_motor(self):
        self.machine.start("continuous", 0.0)
        self.update(0.0)
        self.update(0.1)
        stopped = self.update(0.2)
        self.assertEqual(stopped.stopped_reason, "连续未检测到条纹")
        self.assertIn(("stop", None), stopped.commands)

        self.machine.start("continuous", 1.0)
        disconnected = self.update(1.1, connected=False)
        self.assertEqual(disconnected.stopped_reason, "串口失联")

    def test_maximum_run_time_stops_motor(self):
        self.machine.start("continuous", 0.0)
        stopped = self.update(10.1, color=0.5)
        self.assertEqual(stopped.stopped_reason, "达到最大运行时间")

    def test_step_mode_moves_then_pauses_without_sleep(self):
        self.machine.start("step", 0.0)
        moving = self.update(0.0, color=0.5, params=STEP)
        self.assertEqual(moving.commands, (("set_speed", 5), ("start", None)))
        paused = self.update(0.11, color=0.5, params=STEP)
        self.assertEqual(paused.commands, (("stop", None),))
        self.assertEqual(self.update(0.14, black=0.8, params=STEP).commands, ())


class SerialCommandQueueTests(unittest.TestCase):
    def test_runs_operations_off_caller_thread_and_reports_errors(self):
        commands = SerialCommandQueue("test-serial")
        caller = threading.get_ident()
        commands.submit("thread", threading.get_ident)
        commands.submit("failure", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        deadline = time.monotonic() + 1
        results = []
        while len(results) < 2 and time.monotonic() < deadline:
            results.extend(commands.drain())
            time.sleep(0.005)
        commands.shutdown()
        by_name = {result.name: result for result in results}
        self.assertNotEqual(by_name["thread"].value, caller)
        self.assertIsInstance(by_name["failure"].error, RuntimeError)

    def test_coalesces_duplicate_poll_commands(self):
        commands = SerialCommandQueue("test-coalesce")
        gate = threading.Event()
        self.assertTrue(commands.submit("poll", lambda: gate.wait(0.2), coalesce=True))
        self.assertFalse(commands.submit("poll", lambda: None, coalesce=True))
        gate.set()
        commands.shutdown()


if __name__ == "__main__":
    unittest.main()
