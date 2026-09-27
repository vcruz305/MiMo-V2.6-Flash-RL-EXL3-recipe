"""CPU-only checks: execute the operator wrapper, never a model launcher."""
import os
from pathlib import Path
import subprocess
import shutil
import unittest

SCRIPT = Path(__file__).resolve().with_name('serve-uma-rental.sh')


def run_wrapper(profile=None, dry=True):
    env = dict(os.environ, DRY_RUN='1' if dry else '0')
    env.pop('PROFILE', None)
    if profile is not None:
        env['PROFILE'] = profile
    return subprocess.run([shutil.which('bash'), SCRIPT.name], cwd=SCRIPT.parent, env=env, text=True, capture_output=True)


class OperatorWrapperTests(unittest.TestCase):
    def test_measured_quantized_profiles(self):
        expected = {
            None: 'start_france_quant_dflash4_dynamic06_chunk4096.py',
            'quantized-draft': 'start_france_quant_dflash4_dynamic06_chunk4096.py',
            'quantized-draft-readback': 'round6-quant-readback/start_quant_readback.py',
        }
        for profile, launcher in expected.items():
            with self.subTest(profile=profile):
                result = run_wrapper(profile)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), '/usr/bin/python3 /workspace/mimo-tune/' + launcher)

    def test_missing_readback_dependency_is_explicit(self):
        # Refuse to run this non-dry check on an installed rental.
        probe = subprocess.run([shutil.which('bash'), '-c', 'test ! -e /workspace/mimo-tune'], capture_output=True)
        if probe.returncode:
            self.skipTest('Non-dry failure check requires an unprovisioned machine')
        result = run_wrapper('quantized-draft-readback', dry=False)
        self.assertEqual(result.returncode, 1)
        self.assertIn('isolated Python shadow', result.stderr)
        self.assertIn('no BF16 fallback', result.stderr)

    def test_all_profile_mappings(self):
        import re
        expected = {
            'quantized-draft': 'start_france_quant_dflash4_dynamic06_chunk4096.py',
            'quantized-draft-readback': 'round6-quant-readback/start_quant_readback.py',
            'dynamic-balanced': 'start_france_dflash7_dynamic06_chunk4096.py',
            'chunk1024': 'start_france_dflash7_dynamic06.py',
            'dynamic-code': 'start_france_dflash7_dynamic04.py',
            'dflash4': 'start_france_dflash4_serial.py',
            'dflash4-batch': 'start_france_dflash4_greedy.py',
            'dflash7': 'start_france_dflash7.py',
            'q4-chunk1024': 'start_france_q4_1024.py',
            'baseline': 'start_france_uma.py',
        }
        actual = re.findall(r'^  ([\w-]+)\) launcher=', SCRIPT.read_text(), re.M)
        self.assertEqual(set(actual), set(expected))
        self.assertEqual(len(actual), 10)
        for profile, launcher in expected.items():
            with self.subTest(profile=profile):
                result = run_wrapper(profile)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), '/usr/bin/python3 /workspace/mimo-tune/' + launcher)
        result = run_wrapper('not-a-profile')
        self.assertEqual(result.returncode, 2)
        self.assertIn('Unknown PROFILE', result.stderr)
        self.assertEqual(result.stdout, '')

    def test_dry_run_does_not_require_remote_launcher(self):
        result = run_wrapper('dynamic-balanced')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), '/usr/bin/python3 /workspace/mimo-tune/start_france_dflash7_dynamic06_chunk4096.py')


if __name__ == '__main__':
    unittest.main()
