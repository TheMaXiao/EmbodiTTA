import argparse
import json
import tempfile
import unittest
from pathlib import Path

from emboditta.config import parse_configured_args


class ConfigTests(unittest.TestCase):
    def test_config_values_are_defaults_and_cli_values_override_them(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "config.json"
            config_path.write_text(
                json.dumps({"experiment": {"batch_size": 16, "learning_rate": 0.001}}),
                encoding="utf-8",
            )

            parser = argparse.ArgumentParser()
            parser.add_argument("--batch-size", type=int, default=8)
            parser.add_argument("--learning-rate", type=float, default=0.1)
            args = parse_configured_args(
                parser,
                "experiment",
                ["--config", str(config_path), "--batch-size", "4"],
            )

        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.learning_rate, 0.001)

    def test_missing_config_section_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "config.json"
            config_path.write_text("{}", encoding="utf-8")
            parser = argparse.ArgumentParser()

            with self.assertRaisesRegex(ValueError, "Missing configuration section"):
                parse_configured_args(parser, "missing", ["--config", str(config_path)])


if __name__ == "__main__":
    unittest.main()