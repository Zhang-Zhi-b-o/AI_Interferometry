import unittest
import time

from src.ui.app import (
    _backlash_endpoint_reached,
    _decide_measurement_direction,
)


class TemporaryMeasurementDirectionTests(unittest.TestCase):
    def test_forward_increases_reading_toward_higher_target(self):
        self.assertEqual(
            _decide_measurement_direction(1.200, 1.250, 0.001),
            "forward",
        )

    def test_reverse_decreases_reading_toward_lower_target(self):
        self.assertEqual(
            _decide_measurement_direction(1.300, 1.250, 0.001),
            "reverse",
        )

    def test_stops_inside_tolerance(self):
        self.assertEqual(
            _decide_measurement_direction(1.2505, 1.250, 0.001),
            "stop",
        )

    def test_missing_reading_waits_instead_of_defaulting_forward(self):
        self.assertEqual(
            _decide_measurement_direction(None, 1.250, 0.001),
            "wait",
        )

    def test_motion_command_sets_gear_before_forward_start(self):
        events = []

        class Controller:
            def set_speed(self, gear):
                events.append(("gear", gear))
                return True

            def start_forward(self):
                events.append(("start", "forward"))
                return True

            def start_reverse(self):
                events.append(("start", "reverse"))
                return True

        class Commands:
            def submit(self, _name, operation, **_kwargs):
                self.operation = operation
                return True

        class Log:
            def write(self, _text):
                pass

        from src.ui.app import YoloCamApp
        app = object.__new__(YoloCamApp)
        app._measurement_active = True
        app._measurement_generation = 0
        app._measurement_direction = "stopped"
        app.motor_commands = Commands()
        app.log = Log()

        self.assertTrue(app._queue_measurement_motion(
            "forward", Controller(), 10))
        self.assertTrue(app.motor_commands.operation())
        self.assertEqual(events, [("gear", 10), ("start", "forward")])

    def test_stale_queued_motion_cannot_restart_motor(self):
        events = []

        class Controller:
            def set_speed(self, _gear):
                events.append("gear")
                return True

            def start_forward(self):
                events.append("forward")
                return True

            def start_reverse(self):
                events.append("reverse")
                return True

        class Commands:
            def submit(self, _name, operation, **_kwargs):
                self.operation = operation
                return True

        class Log:
            def write(self, _text):
                pass

        from src.ui.app import YoloCamApp
        app = object.__new__(YoloCamApp)
        app._measurement_active = True
        app._measurement_generation = 0
        app._measurement_direction = "stopped"
        app.motor_commands = Commands()
        app.log = Log()

        app._queue_measurement_motion("reverse", Controller(), 10)
        app._measurement_generation += 1
        self.assertTrue(app.motor_commands.operation())
        self.assertEqual(events, [])

    def test_hardware_control_rejects_displayed_but_stale_reading(self):
        from src.ui.app import YoloCamApp
        app = object.__new__(YoloCamApp)
        app.micrometer_reading_mm = 1.234
        app.micrometer_reading_at = time.time() - 10.0
        self.assertIsNone(app._fresh_micrometer_reading(1.5))
        app.micrometer_reading_at = time.time()
        self.assertEqual(app._fresh_micrometer_reading(1.5), 1.234)

    def test_backlash_never_treats_unchanged_value_as_target(self):
        from src.ui.app import YoloCamApp
        app = object.__new__(YoloCamApp)
        self.assertFalse(app._backlash_at_target(
            2.000, 1.000, tolerance=0.001, reading_timeout=0.1))
        self.assertTrue(app._backlash_at_target(
            2.000, 2.0005, tolerance=0.001, reading_timeout=0.1))

    def test_backlash_forward_endpoint_accepts_near_or_crossed_value(self):
        self.assertFalse(_backlash_endpoint_reached(
            1.980, 2.000, 0.010, "forward"))
        self.assertTrue(_backlash_endpoint_reached(
            1.991, 2.000, 0.010, "forward"))
        self.assertTrue(_backlash_endpoint_reached(
            2.020, 2.000, 0.010, "forward"))

    def test_backlash_reverse_endpoint_accepts_near_or_crossed_value(self):
        self.assertFalse(_backlash_endpoint_reached(
            1.020, 1.000, 0.010, "reverse"))
        self.assertTrue(_backlash_endpoint_reached(
            1.009, 1.000, 0.010, "reverse"))
        self.assertTrue(_backlash_endpoint_reached(
            0.980, 1.000, 0.010, "reverse"))

    def test_backlash_repeated_phase_command_does_not_resubmit_direction(self):
        class Commands:
            def __init__(self):
                self.operations = []

            def submit(self, name, operation, **_kwargs):
                self.operations.append((name, operation))
                return True

        class Controller:
            def set_speed(self, _gear):
                return True

            def start_forward(self):
                return True

            def start_reverse(self):
                return True

        from src.ui.app import YoloCamApp
        app = object.__new__(YoloCamApp)
        app._backlash_active = True
        app._backlash_generation = 0
        app._backlash_motor_direction = "stopped"
        app.motor_commands = Commands()
        controller = Controller()

        app._move_motor(controller, "forward", 10)
        app._move_motor(controller, "forward", 10)
        self.assertEqual(len(app.motor_commands.operations), 1)

    def test_backlash_first_step_can_submit_initial_motion(self):
        class Commands:
            def __init__(self):
                self.names = []

            def submit(self, name, _operation, **_kwargs):
                self.names.append(name)
                return True

        class Root:
            def after(self, _delay, _callback):
                return "job"

        class Panel:
            def set_backlash_status(self, _text):
                pass

        class Log:
            def write(self, _text):
                pass

        class Controller:
            pass

        from src.ui.app import YoloCamApp
        app = object.__new__(YoloCamApp)
        app._backlash_active = True
        app._backlash_phase = "move_to_start"
        app._backlash_started_at = time.monotonic()
        app._backlash_reading_lost_at = 0.0
        app._backlash_start_mm = 1.000
        app._backlash_end_mm = 2.000
        app._backlash_approach_direction = "forward"
        app._backlash_motor_direction = "stopped"
        app._backlash_generation = 0
        app.temporary_measurement_panel = Panel()
        app.motor = Controller()
        app.motor_connected = True
        app.motor_commands = Commands()
        app.root = Root()
        app.log = Log()
        app._fresh_micrometer_reading = lambda _timeout=None: 0.500
        app._update_backlash_center_display = lambda: None

        app._backlash_step()

        self.assertEqual(app._backlash_job, "job")
        self.assertTrue(any(
            name.startswith("backlash_move_")
            for name in app.motor_commands.names))


if __name__ == "__main__":
    unittest.main()
