"""Exercise the actual Streamlit page, including its locked plan and saved record."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from closedloop.menu import menu_test

PAGE = Path(__file__).resolve().parents[1] / "pages" / "5_Run_a_test.py"


def widget(elements, label):
    return next(e for e in elements if e.label == label)


class MenuPageTest(unittest.TestCase):
    def setUp(self):
        config = patch("closedloop.runlog.config", return_value=None)
        config.start()
        self.addCleanup(config.stop)
        self.app = AppTest.from_file(str(PAGE)).run()
        self.assertFalse(self.app.exception)
        self.assertEqual(widget(self.app.selectbox, "Test").value, "real_vs_fakes")
        widget(self.app.selectbox, "Test").select("number_menu").run()

    def test_run_matches_backend_and_keeps_complete_record(self):
        at = self.app
        widget(at.select_slider, "Fakes").set_value(100)
        widget(at.select_slider, "Bar (how far past the fakes)").set_value(99.0)
        widget(at.number_input, "Seed").set_value(78).run()
        self.assertTrue(widget(at.button, "Run").disabled)
        widget(at.button, "Lock").click().run()
        widget(at.button, "Run").click().run()
        self.assertFalse(at.exception)
        out = at.session_state["lab_res"]
        constants = json.loads(at.text_area[0].value)
        expected = menu_test(constants, 0.078, n_fakes=100, seed=78, bar_pct=99.0)
        self.assertEqual(out["result"]["details"], expected)
        self.assertEqual(out["result"]["rows"][0]["real"], expected["distinct_hits"])
        self.assertEqual(out["result"]["rows"][0]["fakes_mean"], expected["fake_mean"])
        self.assertEqual(out["result"]["passed"], expected["passed"])
        self.assertIn("raw hits", " ".join(m.value for m in at.markdown))
        settings = out["plan"]["settings"]
        self.assertEqual(dict(settings["constants"]), constants)
        self.assertEqual(settings["target"], expected["target"])
        self.assertEqual(settings["tol"], expected["settings"]["tol"])
        self.assertEqual(settings["ratios"], expected["settings"]["ratios"])
        self.assertEqual(len(at.dataframe), 2)  # statistics and distinct hit labels/values
        self.assertIn("lattice-matched", json.dumps(out))
        self.assertTrue(at.get("download_button"))

    def test_changing_any_input_invalidates_the_lock(self):
        at = self.app
        changes = [
            lambda: at.text_area[0].set_value('{"a": 10, "b": 128}'),
            lambda: widget(at.number_input, "Target number").set_value(0.0765),
            lambda: widget(at.number_input, "Relative tolerance").set_value(0.02),
            lambda: at.checkbox[0].set_value(False),
            lambda: widget(at.select_slider, "Fakes").set_value(2000),
            lambda: widget(at.number_input, "Seed").set_value(78),
            lambda: widget(at.select_slider, "Bar (how far past the fakes)").set_value(99.0),
        ]
        for change in changes:
            widget(at.button, "Lock").click().run()
            self.assertFalse(widget(at.button, "Run").disabled)
            change().run()
            self.assertTrue(widget(at.button, "Run").disabled)
            self.assertFalse(at.exception)

    def test_invalid_constants_cannot_lock(self):
        at = self.app
        for text in ('{', '{}', '[]', '{"a": 0}', '{"a": -1}', '{"a": NaN}',
                     '{"a": Infinity}', '{"a": "10"}', '{"a": true}', '{"a": 1, "a": 2}'):
            with self.subTest(text=text):
                at.text_area[0].set_value(text).run()
                self.assertFalse(at.exception)
                self.assertTrue(at.error)
                self.assertTrue(widget(at.button, "Lock").disabled)
                self.assertTrue(widget(at.button, "Run").disabled)

    def test_reordering_constants_invalidates_lock_for_seeded_draws(self):
        at = self.app
        constants = json.loads(at.text_area[0].value)
        widget(at.button, "Lock").click().run()
        self.assertFalse(widget(at.button, "Run").disabled)
        at.text_area[0].set_value(json.dumps(dict(reversed(list(constants.items()))))).run()
        self.assertTrue(widget(at.button, "Run").disabled)

    def test_zero_target_or_tolerance_cannot_lock(self):
        at = self.app
        for label in ("Target number", "Relative tolerance"):
            widget(at.number_input, label).set_value(0.0).run()
            self.assertTrue(at.error)
            self.assertTrue(widget(at.button, "Lock").disabled)
            self.assertFalse(at.exception)
            widget(at.number_input, label).set_value(0.01).run()


if __name__ == "__main__":
    unittest.main()
