"""Exercise Unix installers without downloading packages or loading models."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
if not BASH and os.name == "nt":
    candidate = Path("C:/Program Files/Git/bin/bash.exe")
    if candidate.exists():
        BASH = str(candidate)

MOCKS = r'''
uname() { printf '%s\n' "$TEST_OS"; }
sysctl() { printf '%s\n' "$TEST_APPLE"; }
xcode-select() { return "${TEST_XCODE_EXIT:-0}"; }
cc() { return 0; }
c++() { return 0; }
export -f uname sysctl xcode-select cc c++
mock_python() {
  printf '%s | %s\n' "${CMAKE_ARGS:-}" "$*" >> "$TEST_LOG"
  if [[ "$*" == *"print(platform.machine())"* ]]; then
    printf '%s\n' "$TEST_ARCH"
  elif [[ "$*" == "-m venv .venv" ]]; then
    mkdir -p .venv/bin
    printf '#!/bin/bash\nsource "$TEST_MOCKS"\nmock_python "$@"\n' > .venv/bin/python
    chmod +x .venv/bin/python
  elif [[ "$*" == *"--only-binary=:all:"* && "${TEST_WHEEL_FAIL:-0}" == "1" ]]; then
    return 1
  elif [[ "$*" == *"tools/setup_probe.py"* && "${TEST_PROBE_FAIL:-0}" == "1" ]]; then
    return 1
  fi
}
python3.12() { mock_python "$@"; }
export -f mock_python python3.12
'''

@unittest.skipUnless(BASH, "Bash is required for installer flow tests")
class PlatformSetupTests(unittest.TestCase):
    def scenario(self, script, *, system="Darwin", arch="arm64", apple="1",
                 args=(), wheel_fail=False, probe_fail=False, xcode_exit=0,
                 display=None, prepared=False):
        with tempfile.TemporaryDirectory(prefix="aevum setup ") as temp:
            folder = Path(temp)
            for filename in ("macOS-Setup.command", "macOS-Run.command",
                             "Linux-Setup.sh", "Linux-Run.sh"):
                shutil.copyfile(ROOT / filename, folder / filename)
            mocks = folder / "mocks.sh"
            mocks.write_text(MOCKS, encoding="utf-8", newline="\n")
            log = folder / "calls.log"
            env = os.environ.copy()
            for key in ("AEVUM_PYTHON", "DISPLAY", "WAYLAND_DISPLAY"):
                env.pop(key, None)
            env.update(TEST_OS=system, TEST_ARCH=arch, TEST_APPLE=apple,
                       TEST_WHEEL_FAIL=str(int(wheel_fail)),
                       TEST_PROBE_FAIL=str(int(probe_fail)),
                       TEST_XCODE_EXIT=str(xcode_exit),
                       TEST_MOCKS=mocks.as_posix(), TEST_LOG=log.as_posix())
            if display:
                env["DISPLAY"] = display
            prelude = 'source "$TEST_MOCKS"; '
            if prepared:
                prelude += 'mock_python -m venv .venv; '
            result = subprocess.run(
                [BASH, "-c", prelude + 'bash "$@"',
                 "test", script, *args],
                cwd=folder, env=env, text=True, capture_output=True, timeout=30)
            return result, log.read_text() if log.exists() else ""

    def test_apple_silicon_uses_metal(self):
        result, calls = self.scenario("macOS-Setup.command")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("whl/metal", calls)
        self.assertIn("tools/setup_probe.py --require-runtime-deps", calls)

    def test_intel_mac_and_explicit_cpu(self):
        for options in (dict(arch="x86_64", apple="0"), dict(args=("--cpu",))):
            with self.subTest(options=options):
                result, calls = self.scenario("macOS-Setup.command", **options)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("whl/cpu", calls)

    def test_rosetta_is_rejected_before_package_changes(self):
        result, calls = self.scenario("macOS-Setup.command", arch="x86_64")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Rosetta", result.stderr)
        self.assertNotIn("-m pip", calls)

    def test_mac_source_fallback_enables_metal(self):
        result, calls = self.scenario("macOS-Setup.command", wheel_fail=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-DGGML_METAL=ON", calls)
        self.assertIn("--no-binary=llama-cpp-python", calls)

    def test_missing_mac_toolchain_stops_fallback(self):
        result, calls = self.scenario("macOS-Setup.command", wheel_fail=True,
                                      xcode_exit=1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("xcode-select --install", result.stderr)
        self.assertNotIn("--no-binary=llama-cpp-python", calls)

    def test_linux_cpu_install(self):
        result, calls = self.scenario("Linux-Setup.sh", system="Linux",
                                      arch="x86_64", apple="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("whl/cpu", calls)
        self.assertIn("tools/setup_probe.py --require-runtime-deps", calls)

    def test_probe_failure_does_not_report_success(self):
        result, _ = self.scenario("Linux-Setup.sh", system="Linux", probe_fail=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Setup complete", result.stdout)

    def test_linux_source_fallback_disables_gpu_backends(self):
        result, calls = self.scenario("Linux-Setup.sh", system="Linux", wheel_fail=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF", calls)
        self.assertIn("--no-binary=llama-cpp-python", calls)

    def test_linux_launcher_rejects_headless_session(self):
        result, calls = self.scenario("Linux-Run.sh", system="Linux", prepared=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("graphical desktop", result.stderr)
        self.assertNotIn("app.py", calls)

    def test_launchers_use_environment_from_path_with_spaces(self):
        for script, system in (("macOS-Run.command", "Darwin"), ("Linux-Run.sh", "Linux")):
            with self.subTest(script=script):
                result, calls = self.scenario(script, system=system, prepared=True, display=":0")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("app.py", calls)

    def test_wrong_platform_stops_before_install(self):
        result, calls = self.scenario("macOS-Setup.command", system="Linux")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, "")

    def test_launchers_require_setup(self):
        for script, system in (("macOS-Run.command", "Darwin"),
                               ("Linux-Run.sh", "Linux")):
            with self.subTest(script=script):
                result, _ = self.scenario(script, system=system)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("first", result.stderr)

if __name__ == "__main__":
    unittest.main()
