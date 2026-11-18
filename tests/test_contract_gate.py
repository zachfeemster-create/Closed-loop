"""Contract gate must not confuse 'agent ran' with 'feature accepted'."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engineering"))
from contract_gate import evaluate, validate


SPEC = """# Spec: tiny test
## Problem statement
Maintain actual verified test outputs.
## Non-goals
No deployment or scoring.
## Assumptions
Python is present.
## Reference implementation
existing.py is protected.
## Architecture
Independent local checks.
## Test plan
Given an unchanged baseline
When a check executes
Then a machine-readable result must match.
"""


class SpecContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "spec.md").write_text(SPEC)
        (self.root / "existing.py").write_text("print('alive')\n")
        (self.root / "result.json").write_text('{"status": "UNVERIFIED", "false_accepts": 0}\n')
        self.contract = {
            "schema": "closedloop.acceptance-contract.v1",
            "id": "test-feature",
            "spec": "spec.md",
            "checks": [
                {"name": "real-code-executed", "kind": "command",
                 "argv": ["python", "existing.py"], "timeout_seconds": 10},
                {"name": "verify-artifact", "kind": "json_assert",
                 "path": "result.json",
                 "equals": {"status": "UNVERIFIED", "false_accepts": 0}},
            ],
        }
        self.path = self.root / "contract.json"
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.contract))

    def test_real_checks_run_and_are_source_hashed(self):
        r = evaluate(self.root, self.path)
        self.assertEqual(r["status"], "VERIFIED_SPEC_CHECKS")
        self.assertEqual(r["checks_executed"], 2)
        self.assertEqual(len(r["spec_sha256"]), 64)
        self.assertIn("alive", r["results"][0]["stdout_tail"])
        self.assertIn("not a domain-level verifier", r["next_action"])

    def test_bad_output_is_rejected_not_green(self):
        self.contract["checks"][1]["equals"]["false_accepts"] = 1
        self.save()
        r = evaluate(self.root, self.path)
        self.assertFalse(r["accepted"])
        self.assertEqual(r["status"], "REJECTED_BY_INDEPENDENT_CHECK")
        self.assertEqual(r["results"][-1]["observed"]["false_accepts"], 0)

    def test_first_failure_stops_loop_and_preserves_feedback(self):
        self.contract["checks"][0]["argv"] = ["python", "missing.py"]
        self.save()
        r = evaluate(self.root, self.path)
        self.assertFalse(r["accepted"])
        self.assertEqual(r["checks_executed"], 1)
        self.assertEqual(r["results"][0]["exit_code"], 2)

    def test_no_spec_without_given_when_then_or_six_sections(self):
        (self.root / "spec.md").write_text(SPEC.replace("## Test plan", "## Wish list"))
        with self.assertRaisesRegex(ValueError, "six required"):
            validate(self.root, self.contract)
        (self.root / "spec.md").write_text(SPEC.replace("Then a machine", "Could a machine"))
        with self.assertRaisesRegex(ValueError, "Given/When/Then"):
            validate(self.root, self.contract)

    def test_no_shell_no_package_download_no_directory_escape(self):
        for argv in (
            ["bash", "do-anything.sh"],
            ["python", "-c", "print('fake pass')"],
            ["python", "-m", "pip", "install", "x"],
            ["python", "../secrets.py"],
        ):
            self.contract["checks"][0]["argv"] = argv
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                validate(self.root, self.contract)

    def test_no_unbounded_or_missing_test_commands(self):
        self.contract["checks"][0]["timeout_seconds"] = 100000
        with self.assertRaisesRegex(ValueError, "timeout"):
            validate(self.root, self.contract)
        self.contract["checks"] = []
        with self.assertRaisesRegex(ValueError, "Require 1..8"):
            validate(self.root, self.contract)

    def test_existing_json_is_required_not_claimed(self):
        (self.root / "result.json").unlink()
        r = evaluate(self.root, self.path)
        self.assertFalse(r["accepted"])
        self.assertEqual(r["results"][-1]["kind"], "json_assert")
        self.assertIn("FileNotFoundError", r["results"][-1]["error"])


    def test_unittest_p_is_a_filename_pattern_not_python_eval(self):
        self.contract["checks"][0]["argv"] = [
            "python", "-m", "unittest", "discover", "-s", "tests",
            "-p", "test_energy_constraints.py", "-v",
        ]
        self.assertEqual(len(validate(self.root, self.contract)[1]), 2)
        self.contract["checks"][0]["argv"] = ["node", "-p", "process.env"]
        with self.assertRaisesRegex(ValueError, "Interactive evaluation"):
            validate(self.root, self.contract)

if __name__ == "__main__":
    unittest.main()
