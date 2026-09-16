import copy
import json
import unittest
from pathlib import Path

from generate_synthetic_dataset import (
    DEFAULT_CONFIG,
    generate_variant,
    parse_database_text,
)
from infer import finalize_report, prepare_analysis_input
from input_normalization import build_analysis_input, normalize_runtime_input
from policy_retrieval import PolicyRetriever
from prepare_dataset import expand_input_records, family_key, validate_row
from schema import validate_report
from telemetry import validate_snapshot_shape


ROOT = Path(__file__).resolve().parents[1]


def load_json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def policy_document(url: str = "https://x.com/en/privacy") -> dict:
    return {
        "url": url,
        "found": True,
        "applicable": True,
        "sections": [
            {
                "heading": "Information We Collect",
                "text": (
                    "We collect browser and device information and use cookies "
                    "and similar technologies when people use the service."
                ),
            }
        ],
        "limitations": [],
    }


class FakeRetriever:
    def __init__(self, document: dict):
        self.document = document
        self.calls = []

    def retrieve(self, *, domain_url, privacy_policy_url=None):
        self.calls.append((domain_url, privacy_policy_url))
        return copy.deepcopy(self.document)


class RuntimeContractTests(unittest.TestCase):
    def test_exact_connector_shape_is_accepted(self):
        record = load_json("examples/inference_input.json")
        runtime = normalize_runtime_input(record)

        self.assertEqual(runtime["domain_url"], "https://x.com")
        self.assertEqual(runtime["visit"]["duration_seconds"], 1)
        self.assertEqual(runtime["visit"]["observed_at"], "2026-09-08T13:09:13.702000Z")
        self.assertEqual(runtime["snapshot"]["observation"]["totalRequests"], 29)
        self.assertEqual(runtime["snapshot"]["signals"][0]["api"], "Navigator")

        analysis_input = build_analysis_input(runtime, policy_document())
        self.assertEqual(
            set(analysis_input),
            {
                "domain_url",
                "privacy_policy_url",
                "visit",
                "seen_behavior",
                "thirdPartyHosts",
                "trackers",
                "signals",
                "page",
                "detections",
                "policy_document",
            },
        )
        self.assertEqual(
            analysis_input["seen_behavior"],
            {
                "observations": {
                    "totalRequests": 29,
                    "firstPartyRequests": 6,
                    "thirdPartyRequests": 23,
                }
            },
        )
        self.assertNotIn("security", analysis_input)
        self.assertNotIn("interest", analysis_input)

    def test_upload_envelope_uses_top_level_created_at(self):
        snapshot = load_json("examples/virustotal_snapshot.json")
        runtime = normalize_runtime_input(
            {
                "snapshotId": "outer-snapshot-id",
                "createdAt": 1788872953702,
                "payload": snapshot,
                "domain_url": "https://www.virustotal.com",
                "privacy_policy_url": None,
            }
        )
        self.assertEqual(runtime["visit"]["snapshot_id"], "outer-snapshot-id")
        self.assertEqual(runtime["visit"]["observed_at"], "2026-09-08T13:09:13.702000Z")

    def test_bad_supplied_policy_url_falls_through_to_retriever(self):
        record = load_json("examples/inference_input.json")
        record["privacy_policy_url"] = "file:///tmp/not-a-policy"
        fake = FakeRetriever(policy_document())
        prepared = prepare_analysis_input(record, retriever=fake)

        self.assertEqual(fake.calls, [("https://x.com", "file:///tmp/not-a-policy")])
        self.assertTrue(prepared["policy_document"]["applicable"])

    def test_old_synthetic_aliases_are_rejected(self):
        snapshot = load_json("examples/virustotal_snapshot.json")
        snapshot["thirdPartyHosts"][0]["types"] = snapshot["thirdPartyHosts"][0].pop(
            "resourceTypes"
        )
        with self.assertRaisesRegex(ValueError, "resourceTypes"):
            validate_snapshot_shape(snapshot)


class RetrievalTests(unittest.TestCase):
    def test_policy_content_detection_and_cross_site_applicability(self):
        text = (
            "Privacy Notice. Information we collect includes personal information "
            "and device information. We explain how we use and share it, your privacy "
            "rights, retention choices, and data protection controls. " * 5
        )
        self.assertTrue(
            PolicyRetriever._looks_like_policy(
                "https://legal.example/privacy", "Example Privacy Notice", text
            )
        )
        self.assertTrue(
            PolicyRetriever._is_applicable(
                domain_url="https://virustotal.com",
                policy_url="https://cloud.example/privacy",
                method="web_search",
                title="VirusTotal Privacy Notice",
                body_text=text,
            )
        )
        self.assertFalse(
            PolicyRetriever._is_applicable(
                domain_url="https://virustotal.com",
                policy_url="https://unrelated.example/privacy",
                method="domain_link",
                title="Unrelated Service Privacy Notice",
                body_text=text,
            )
        )

    def test_private_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "local or internal"):
            PolicyRetriever()._assert_safe_url("http://localhost/privacy")


class OutputAndGeneratorTests(unittest.TestCase):
    def test_mongosh_database_export_is_parsed_without_code_execution(self):
        export = r'''
        [
          {
            _id: ObjectId('64b000000000000000000001'),
            policy_id: 'ppa-one',
            policy_raw_results: {
              domain: 'https://one.example',
              findings: [{category: 'cookies', behavior: 'Cookie access'}]
            },
          }
        ]
        Type "it" for more
        veilance_test> it
        [
          {
            _id: ObjectId("64b000000000000000000002"),
            policy_id: 'ppa-two',
            policy_raw_results: {
              domain: 'https://two.example',
              findings: [{category: 'telemetry', behavior: 'Beacon reporting'}]
            }
          }
        ]
        '''
        records = parse_database_text(export)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["_id"], "64b000000000000000000001")
        self.assertEqual(
            records[1]["policy_raw_results"]["findings"][0]["category"],
            "telemetry",
        )

    def test_prepare_can_expand_processed_database_records(self):
        report = load_json("examples/virustotal_expected_report.json")
        records = [
            {
                "_id": "database-row-one",
                "policy_id": "ppa-one",
                "policy_raw_results": report,
                "associated_domain": report["domain"],
                "associated_policy_link": report["privacy_policy"]["url"],
            }
        ]
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["no_policy_probability"] = 0.0
        config["augment_probability"] = 0.0
        rows, stats = expand_input_records(
            records,
            input_format="database",
            synthetic_variants=2,
            generator_seed="database-unit-test",
            generator_config=config,
        )
        self.assertEqual(stats["database_source_records"], 1)
        self.assertEqual(stats["synthetic_rows"], 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[0]["_synthetic_metadata"]["provenance"]["database_record_id"],
            "database-row-one",
        )
        self.assertEqual(
            rows[0]["_synthetic_metadata"]["family_id"],
            rows[1]["_synthetic_metadata"]["family_id"],
        )
        for index, row in enumerate(rows):
            validate_row(row, index, 24_000)
        direct_row = load_json("examples/raw_training_row.json")
        self.assertEqual(family_key(rows[0]), family_key(direct_row))

    def test_checked_in_expected_report_is_valid(self):
        validate_report(load_json("examples/virustotal_expected_report.json"))

    def test_report_validator_rejects_markdown_and_bad_counts(self):
        report = load_json("examples/virustotal_expected_report.json")
        report["analysis"]["summary"] = "**Unsupported formatting**"
        with self.assertRaisesRegex(ValueError, "Markdown formatting"):
            validate_report(report)

        report = load_json("examples/virustotal_expected_report.json")
        report["analysis"]["counts"]["matched"] += 1
        with self.assertRaisesRegex(ValueError, "counts mismatch"):
            validate_report(report)

    def test_generator_produces_coherent_trainable_row(self):
        source = load_json("examples/virustotal_expected_report.json")
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["no_policy_probability"] = 0.0
        config["augment_probability"] = 0.0
        row = generate_variant(source, 0, 0, "unit-test", config)

        validate_snapshot_shape(row["telemetry"])
        validate_report(row["expected"])
        prepared = validate_row(row, 0, 24_000)
        self.assertEqual(
            prepared["seen_behavior"]["observations"]["totalRequests"],
            row["telemetry"]["observation"]["totalRequests"],
        )
        self.assertEqual(prepared["policy_document"]["url"], row["policy_document"]["url"])

    def test_finalize_rejects_hallucinated_policy_section(self):
        source = load_json("examples/virustotal_expected_report.json")
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["no_policy_probability"] = 0.0
        config["augment_probability"] = 0.0
        row = generate_variant(source, 0, 1, "grounding-test", config)
        analysis_input = validate_row(row, 0, 24_000)
        report = copy.deepcopy(row["expected"])
        disclosed = next(
            (
                finding
                for finding in report["findings"]
                if finding["policy"]["status"]
                in {
                    "explicitly_disclosed",
                    "broadly_disclosed",
                    "implicitly_disclosed",
                    "contradicted",
                }
            ),
            None,
        )
        if disclosed is None:
            self.skipTest("deterministic scenario contained no disclosed finding")
        disclosed["policy"]["section"] = "Invented Policy Section"
        with self.assertRaisesRegex(ValueError, "not extracted by Playwright"):
            finalize_report(report, analysis_input)

    def test_finalize_neutralizes_claims_when_no_policy_is_available(self):
        source = load_json("examples/virustotal_expected_report.json")
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["no_policy_probability"] = 1.0
        config["augment_probability"] = 0.0
        row = generate_variant(source, 0, 2, "no-policy-test", config)
        analysis_input = validate_row(row, 0, 24_000)

        # Simulate an untrusted model attempting a stronger policy conclusion.
        report = copy.deepcopy(row["expected"])
        report["analysis"]["summary"] = "The site contradicted its policy."
        for finding in report["findings"]:
            finding["policy"] = {
                "status": "contradicted",
                "evidence": "Invented policy text.",
                "section": "Invented section",
            }
            finding["comparison"] = "possible_contradiction"
            finding["severity"] = "critical"

        finalized = finalize_report(report, analysis_input)
        self.assertEqual(
            finalized["privacy_policy"],
            {"url": "", "found": False, "applicable": False},
        )
        self.assertNotIn("contradicted", finalized["analysis"]["summary"].lower())
        self.assertTrue(
            all(item["comparison"] == "indeterminate" for item in finalized["findings"])
        )
        self.assertTrue(
            all(item["severity"] == "informational" for item in finalized["findings"])
        )


if __name__ == "__main__":
    unittest.main()
