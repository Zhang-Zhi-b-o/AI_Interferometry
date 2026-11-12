"""Port changes must trigger device-only setup, without repeatedly elevating."""
from contextlib import contextmanager
import unittest
from unittest.mock import Mock, patch

from app.engine.hardware import signal_setup as setup
from app.services.reading import ReadingService
from app.engine.hardware.signal_device import Device


@contextmanager
def inventory(filters=None, count=1):
    yield None, None, [(None, setup.TARGET + str(i), filters or []) for i in range(count)]


class SignalSetupTests(unittest.TestCase):
    def test_attached_driver_without_input_channels_requests_windows_restart(self):
        device = Device.__new__(Device)
        device.k = Mock()
        device.k.CreateFileW.side_effect = range(1, 11)
        device.handle = device.event = None
        error = OSError('unbound control channel')
        error.winerror = 1
        device.ioctl = Mock(side_effect=error)
        with patch.object(setup, 'ensure_configured'):
            with self.assertRaisesRegex(RuntimeError, '重启 Windows'):
                device.open()
        self.assertEqual(device.k.CloseHandle.call_count, 10)
        self.assertIsNone(device.handle)

    def test_configured_port_does_not_elevate(self):
        with patch.object(setup, 'check_class_filters'), \
             patch.object(setup, 'target_devices', return_value=inventory(['keyboard'])), \
             patch.object(setup.Path, 'is_file', return_value=True), \
             patch.object(setup.subprocess, 'run') as run:
            setup.ensure_configured()
        run.assert_not_called()

    def test_new_port_is_configured_and_reenumerated(self):
        with patch.object(setup, 'check_class_filters'), \
             patch.object(setup, 'target_devices', side_effect=[inventory(), inventory(['keyboard'])]), \
             patch.object(setup.c.windll.shell32, 'IsUserAnAdmin', return_value=0), \
             patch.object(setup.subprocess, 'run', return_value=Mock(returncode=0)) as run:
            setup.ensure_configured()
        command = run.call_args.args[0][-1]
        self.assertIn('-Verb RunAs', command)
        self.assertIn('-Wait -PassThru', command)
        self.assertIn('--configure', command)
        self.assertNotIn('/install', command)

    def test_missing_driver_triggers_repair_even_on_known_port(self):
        with patch.object(setup, 'check_class_filters'), \
             patch.object(setup, 'target_devices', return_value=inventory(['keyboard'])), \
             patch.object(setup.Path, 'is_file', return_value=False), \
             patch.object(setup.c.windll.shell32, 'IsUserAnAdmin', return_value=1), \
             patch.object(setup, 'configure') as configure:
            setup.ensure_configured()
        configure.assert_called_once_with()

    def test_missing_or_ambiguous_device_does_not_elevate(self):
        for count in (0, 2):
            with self.subTest(count=count), patch.object(setup, 'check_class_filters'), \
                 patch.object(setup, 'target_devices', return_value=inventory(count=count)), \
                 patch.object(setup.subprocess, 'run') as run:
                with self.assertRaises(RuntimeError):
                    setup.ensure_configured()
                run.assert_not_called()

    def test_failed_uac_is_reported(self):
        with patch.object(setup, 'check_class_filters'), \
             patch.object(setup, 'target_devices', return_value=inventory()), \
             patch.object(setup.c.windll.shell32, 'IsUserAnAdmin', return_value=0), \
             patch.object(setup.subprocess, 'run', return_value=Mock(returncode=1)):
            with self.assertRaisesRegex(RuntimeError, '管理员确认'):
                setup.ensure_configured()

    def test_connection_failure_remains_available_until_stop(self):
        service = ReadingService()
        with patch('app.engine.hardware.signal_micrometer.SignalMicrometerReader') as reader:
            reader.return_value.connect.side_effect = RuntimeError('USB setup failed')
            ok, message = service.start(source='signal')
        self.assertFalse(ok)
        self.assertEqual(service.last_error, message)
        service.stop()
        self.assertEqual(service.last_error, '')


if __name__ == '__main__':
    unittest.main()
