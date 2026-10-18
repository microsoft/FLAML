import unittest
from importlib.metadata import requires

from packaging.requirements import Requirement


class TestPackaging(unittest.TestCase):
    def test_pyarrow_test_dependency(self):
        requirements = [Requirement(requirement) for requirement in requires("FLAML")]
        for python_version in ("3.10", "3.11", "3.12", "3.13"):
            for sys_platform, platform_machine in (
                ("win32", "ARM64"),
                ("win32", "AMD64"),
                ("linux", "x86_64"),
                ("linux", "aarch64"),
                ("darwin", "arm64"),
            ):
                with self.subTest(
                    python_version=python_version, sys_platform=sys_platform, platform_machine=platform_machine
                ):
                    environment = {
                        "extra": "test",
                        "python_version": python_version,
                        "sys_platform": sys_platform,
                        "platform_machine": platform_machine,
                    }
                    active = {
                        requirement.name: requirement
                        for requirement in requirements
                        if requirement.marker is None or requirement.marker.evaluate(environment)
                    }
                    windows_arm64 = sys_platform == "win32" and platform_machine == "ARM64"
                    if windows_arm64 or python_version == "3.13":
                        self.assertNotIn("pyarrow", active)
                    else:
                        self.assertIn("pyarrow", active)
                        self.assertIn("25.0.0", active["pyarrow"].specifier)
                        self.assertNotIn("26.0.0", active["pyarrow"].specifier)
                    if windows_arm64:
                        for dependency in ("openml", "datasets", "evaluate"):
                            self.assertNotIn(dependency, active)


if __name__ == "__main__":
    unittest.main()
