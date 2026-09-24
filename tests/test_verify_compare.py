"""Observable paired-score accounting and distribution diagnostics."""

import math
import json
import struct
import tempfile
import unittest
from pathlib import Path

from tools.ppl.verify_campaign import record_success
from tools.ppl.verify_compare import campaign_report, compare, logit_comparison, read_scores


class VerifyCompareTests(unittest.TestCase):
    def test_report_marks_missing_smoke_and_required_captures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # TSV-only checkpoints must not imply that distribution evidence exists.
            for profile in "ac":
                (root / f"phase-c-{profile}-common-p8.tsv").write_text(
                    "document\tposition\ttoken\tnll\n0\t9\t42\t1\n")
            result = campaign_report(root)
            self.assertIn("A/c1-w5-8k", result["missing_cells"])
            self.assertIn("C/c4-w5-8k", result["missing_cells"])
            self.assertIn("A/common-p8", result["missing_captures"])
            self.assertIn("C/common-p8", result["missing_captures"])

    def test_filtered_rerun_retains_prior_successful_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "commands.json"
            record_success(path, {"profile": "A", "cell": "one", "command": ["first"]})
            record_success(path, {"profile": "A", "cell": "two", "command": ["second"]})
            record_success(path, {"profile": "A", "cell": "two", "command": ["rerun"]})
            entries = json.loads(path.read_text())
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0]["command"], ["first"])
            self.assertEqual(entries[1]["command"], ["rerun"])

    def test_weighted_pair_and_block_interval(self):
        a = {(0, i): (i, 1.0) for i in range(5)}
        b = {(0, i): (i, 1.25) for i in range(5)}
        result = compare(a, b, block_size=2)
        self.assertEqual(result["tokens"], 5)
        self.assertEqual(result["blocks"], 3)
        self.assertEqual(result["delta_nll"], .25)
        self.assertEqual(result["block_bootstrap_95_ci"], [.25, .25])
        self.assertAlmostEqual(result["ppl_ratio"], math.exp(.25))

    def test_mismatched_gold_or_position_is_rejected(self):
        with self.assertRaises(ValueError):
            compare({(0, 1): (3, 1.0)}, {(0, 1): (4, 1.0)})
        with self.assertRaises(ValueError):
            compare({(0, 1): (3, 1.0)}, {(0, 2): (3, 1.0)})

    def test_context_trends_do_not_cross_document_resets(self):
        a = {(doc, pos): (pos, 1.0) for doc, length in ((0, 1100), (1, 100))
             for pos in range(9, 9 + length)}
        b = {key: (token, nll + (.25 if key[0] == 0 else .5))
             for key, (token, nll) in a.items()}
        result = compare(a, b)
        trends = result["position_trends"]
        self.assertEqual([v["tokens"] for v in trends], [1024, 76, 100])
        self.assertTrue(all(v["first"][0] == v["last"][0] for v in trends))
        self.assertEqual([v["delta_nll"] for v in trends], [.25, .25, .5])

    def test_replicated_lanes_are_counted_once_and_must_be_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.tsv"
            path.write_text("document\tposition\ttoken\tnll\n0\t9\t42\t1\n1\t9\t42\t1\n")
            scores = read_scores(path, replicated_lanes=True)
            self.assertEqual(len(scores), 1)
            path.write_text("document\tposition\ttoken\tnll\n0\t9\t42\t1\n1\t9\t42\t1.1\n")
            with self.assertRaises(ValueError):
                read_scores(path, replicated_lanes=True)

    def test_distribution_metrics_use_normalized_full_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a", Path(directory) / "b"
            # A=[0,0], B=[0,1], represented exactly in BF16.
            header = struct.pack("<4I", 2, 0, 9, 0)
            a.write_bytes(header + struct.pack("<2H", 0, 0))
            b.write_bytes(header + struct.pack("<2H", 0, 0x3F80))
            result = logit_comparison(a, b)[0]
            self.assertAlmostEqual(result["delta_nll"], math.log1p(math.e) - math.log(2))
            self.assertAlmostEqual(result["total_variation"], .5 - 1 / (1 + math.e))
            self.assertAlmostEqual(result["kl_baseline_candidate"], math.log1p(math.e) - .5 - math.log(2))
            self.assertEqual(result["candidate_top"], 1)
            self.assertEqual(result["candidate_margin"], 1)


if __name__ == "__main__":
    unittest.main()
