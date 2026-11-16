"""Independent prompt-data oracles and the NIAH scorer's observable report contract."""

from __future__ import annotations

import contextlib
import io
import json
import re
import tempfile
import unittest
import urllib.error
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from tools.bench import make_niah_positions as generator
from tools.bench import run_niah_check as runner


class NiahTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.master = generator.extract_master(
            runner.ROOT / "examples/cli/messages/long_niah_256k.json"
        )

    def test_generated_answers_against_prompt_data(self) -> None:
        for suite in generator.HARD_SUITES:
            for length in ("32k", "128k", "260k"):
                for position in generator.POS_FRACTION:
                    with self.subTest(suite=suite, length=length, layout=position):
                        fixture = generator.build_hard_fixture(self.master, length, position, suite)
                        text = fixture["messages"][1]["content"]
                        document, question = text.split("</document>\n\n")
                        metadata = fixture["niah"]
                        if suite in ("multi-value", "multi-query"):
                            entries = re.findall(r"Registry entry: (\w+) -> (\d+)\.", document)
                            self.assertEqual(len(entries), 24)
                            if suite == "multi-value":
                                key = re.search(r"associated with (\w+)", question).group(1)
                                expected = {value for name, value in entries if name == key}
                                self.assertEqual(len(expected), 6)
                            else:
                                keys = re.findall(r"KEY_[A-Z]+", question)
                                expected = {key: dict(entries)[key] for key in keys}
                                self.assertEqual(len(expected), 4)
                        elif suite.startswith("variable-tracking"):
                            chains, chain_length = {
                                "variable-tracking-1x3": (1, 3),
                                "variable-tracking-1x6": (1, 6),
                                "variable-tracking-2x6": (2, 6),
                                "variable-tracking": (4, 6),
                            }[suite]
                            assignments = re.findall(r"Assignment: (X\d+)=(X\d+|\d+);", document)
                            self.assertEqual(len(assignments), chains * chain_length)
                            state = {}
                            for name, source in assignments:
                                state[name] = state[source] if source.startswith("X") else source
                            self.assertEqual(len(state), chains * chain_length)
                            self.assertEqual(len(set(state.values())), chains)
                            # Round-robin interleaving must preserve each chain's dependency order.
                            for wave in range(1, chain_length):
                                for chain in range(chains):
                                    self.assertEqual(
                                        assignments[wave * chains + chain][1],
                                        assignments[(wave - 1) * chains + chain][0],
                                    )
                            target = re.search(r"final value (\d+)", question).group(1)
                            expected = {name for name, value in state.items() if value == target}
                            self.assertEqual(len(expected), chain_length)
                            self.assertEqual(
                                fixture,
                                generator.build_hard_fixture(self.master, length, position, suite),
                            )
                        elif suite == "aggregation":
                            fragments = re.findall(
                                r"Word list fragment \d+: \[([^\]]+)\]", document
                            )
                            self.assertEqual(len(fragments), 16)
                            counts = Counter(
                                word for part in fragments for word in part.split(", ")
                            )
                            self.assertEqual(sum(counts.values()), 318)
                            ranked = sorted(counts, key=lambda word: counts[word], reverse=True)
                            self.assertGreater(counts[ranked[2]], counts[ranked[3]])
                            expected = set(ranked[:3])
                        else:
                            name = re.search(r"founder of (\w+) choose", question).group(1)
                            self.assertIn(f"founder of {name} told me", document)
                            expected = {
                                re.search(r"use (\d+) as the access code", document).group(1)
                            }
                            self.assertNotIn("OFFICIAL RECORD", document)
                            self.assertNotIn("Registry entry", document)
                        if isinstance(expected, set):
                            self.assertEqual(expected, set(metadata["expected"]))
                            expected = sorted(expected, reverse=True)
                            answers = expected
                        else:
                            self.assertEqual(expected, metadata["expected"])
                            answers = list(expected.values())
                        for answer in answers:
                            self.assertNotIn(answer, question)
                        score = runner.score_answer(
                            json.dumps(expected), metadata["expected"], metadata["scoring"]
                        )
                        self.assertTrue(score["passed"])
                        self.assertEqual(score["matched"], len(expected))

    def test_ladder_cli_generation_and_scoring(self) -> None:
        for suite in (
            "variable-tracking-1x3",
            "variable-tracking-1x6",
            "variable-tracking-2x6",
            "variable-tracking",
        ):
            with self.subTest(suite=suite), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                generate = [
                    "make_niah_positions",
                    "--suite",
                    suite,
                    "--length",
                    "32k",
                    "--position",
                    "all",
                    "--out-dir",
                    str(root),
                ]
                with patch("sys.argv", generate), contextlib.redirect_stdout(io.StringIO()):
                    generator.main()
                calls = []

                def post(base, key, body, timeout, suite=suite, root=root, calls=calls):
                    cases = runner.matrix_cases(
                        ["32k"], runner.NIAH_POSITIONS, suite=suite, fixture_dir=root
                    )
                    fixture = json.loads(Path(cases[len(calls)][1]).read_text())
                    self.assertEqual(body["messages"], fixture["messages"])
                    self.assertNotIn("niah", body)
                    calls.append(body)
                    answer = json.dumps(list(reversed(fixture["niah"]["expected"])))
                    return {"choices": [{"message": {"content": answer}}]}

                run = [
                    "run_niah_check",
                    "--suite",
                    suite,
                    "--lengths",
                    "32k",
                    "--key",
                    "test",
                    "--fixture-dir",
                    str(root),
                    "--output-dir",
                    str(root / "results"),
                ]
                with (
                    patch("sys.argv", run),
                    patch.object(runner, "post", post),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(runner.main(), 0)
                self.assertEqual(len(calls), 5)

    def test_structured_scoring_requires_all_values_and_bindings(self) -> None:
        target = ["alpha", "beta", "gamma"]
        for answer in (
            '["alpha","beta"]',
            '["alpha","beta","gamma","extra"]',
            '["alpha","beta","gamma","gamma"]',
            '```json\n["alpha","beta","gamma"]\n```',
            '["alpha","beta",3]',
            "null",
        ):
            self.assertFalse(runner.score_answer(answer, target, "json-set")["passed"])
        target = {"A": "123", "B": "456"}
        self.assertTrue(
            runner.score_answer('{"B":"456","A":"123"}', target, "json-object")["passed"]
        )
        for answer in (
            '{"A":"456","B":"123"}',
            '{"A":"123","B":"456","C":"789"}',
            '{"A":"wrong","A":"123","B":"456"}',
        ):
            self.assertFalse(runner.score_answer(answer, target, "json-object")["passed"])

    def test_cli_persists_full_answers_and_request_errors_fail_the_gate(self) -> None:
        fixture = generator.build_hard_fixture(self.master, "32k", "mid", "multi-value")
        expected = fixture["niah"]["expected"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = root / "prompt.json"
            prompt.write_text(json.dumps(fixture), encoding="utf-8")
            for label, answer, status in (
                ("pass", expected, 0),
                ("missing", expected[:-1], 1),
                ("error", None, 2),
            ):
                with self.subTest(label=label):
                    response = {"choices": [{"message": {"content": json.dumps(answer)}}]}
                    argv = [
                        "run_niah_check",
                        "--fixture",
                        str(prompt),
                        "--key",
                        "test",
                        "--output-dir",
                        str(root / label),
                        "--label",
                        label,
                    ]

                    def post(base, key, body, timeout, answer=answer, response=response):
                        self.assertEqual(body["messages"], fixture["messages"])
                        self.assertNotIn("niah", body)
                        if answer is None:
                            raise urllib.error.HTTPError(
                                base, 400, "bad request", {}, io.BytesIO(b"test error")
                            )
                        return response

                    with (
                        patch("sys.argv", argv),
                        patch.object(runner, "post", post),
                        contextlib.redirect_stdout(io.StringIO()),
                    ):
                        self.assertEqual(runner.main(), status)
                    paths = list((root / label).glob("*.json"))
                    self.assertEqual(len(paths), 1)
                    record = json.loads(paths[0].read_text())
                    case = record["cases"][0]
                    self.assertEqual(case["total"], 1)
                    self.assertEqual(case["passed"], status == 0)
                    self.assertEqual(case["recall"], 1 if status == 0 else 0)
                    if status == 2:
                        self.assertIn("test error", case["trials"][0]["error"])
                        self.assertEqual(record["errors"], 1)
                    else:
                        self.assertEqual(case["trials"][0]["response"], response)
                        self.assertEqual(case["trials"][0]["answer"], json.dumps(answer))


if __name__ == "__main__":
    unittest.main()
