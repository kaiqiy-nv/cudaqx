#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise playback CLI argument handling and CSV publication."""

import argparse
import csv
import io
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

EMULATOR = None
NOISE_KEYS = (
    "before_measure_flip_probability",
    "after_clifford_depolarization",
    "before_round_data_depolarization",
    "after_reset_flip_probability",
)


class PlaybackCliTest(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="playback-cli-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "config.yaml"
        self.config.write_text("""\
decoders:
  - id: 0
    type: single_error_lut
    block_size: 1
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1]
""")
        self.schedule = self.root / "schedule.txt"
        self.schedule.write_text("0 reset\n")

    def invoke(self, *args, stdout=subprocess.PIPE):
        return subprocess.run([
            EMULATOR, f"--config={self.config}", f"--schedule={self.schedule}",
            "--backend=null", *args
        ],
                              text=True,
                              stdout=stdout,
                              stderr=subprocess.PIPE,
                              timeout=15)

    def assert_reset_csv(self, text):
        records = list(csv.DictReader(io.StringIO(text)))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["op"], "reset")
        self.assertEqual(records[0]["status"], "0")
        self.assertEqual(records[0]["dispatched"], "1")

    def test_help(self):
        result = subprocess.run([EMULATOR, "--help"],
                                text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Usage: playback-emulator --schedule=<file>",
                      result.stdout)

    def test_missing_schedule(self):
        result = subprocess.run([EMULATOR],
                                text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                timeout=15)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Usage: playback-emulator --schedule=<file>",
                      result.stdout)

    def test_unknown_option(self):
        result = self.invoke("--unknown-option")
        self.assertEqual(result.returncode, 1)
        self.assertIn("unrecognized argument: --unknown-option",
                      result.stderr)

    def test_udp_backend_requires_endpoint_for_config_decoder(self):
        result = self.invoke("--backend=udp")
        self.assertEqual(result.returncode, 1)
        self.assertIn("--backend=udp: decoder 0 in --config has no "
                      "--*-endpoint=", result.stderr)

    def test_cpu_roce_backend_requires_endpoint_for_config_decoder(self):
        result = self.invoke("--backend=cpu_roce")
        self.assertEqual(result.returncode, 1)
        self.assertIn("--backend=cpu_roce: decoder 0 in --config has no "
                      "--*-endpoint=", result.stderr)

    def test_inproc_backend_dispatches_fixture_decoder(self):
        result = self.invoke("--backend=inproc")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_reset_csv(result.stdout)

    def test_inproc_backend_filters_endpoint_claimed_decoder(self):
        decoder = """\
    type: single_error_lut
    block_size: 1
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1]
"""
        self.config.write_text("decoders:\n"
                               "  - id: 0\n" + decoder +
                               "  - id: 1\n" + decoder)
        self.schedule.write_text("0 reset session=1\n")

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as endpoint:
            endpoint.bind(("127.0.0.1", 0))
            port = endpoint.getsockname()[1]
            result = self.invoke(
                "--backend=inproc",
                f"--udp-endpoint=0:127.0.0.1:{port}")

        self.assertEqual(result.returncode, 0, result.stderr)
        records = list(csv.DictReader(io.StringIO(result.stdout)))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["op"], "reset")
        self.assertEqual(records[0]["decoder_id"], "1")
        self.assertEqual(records[0]["status"], "0")
        self.assertEqual(records[0]["dispatched"], "1")

    def test_inproc_endpoint_claim_preserves_static_source_stream(self):
        help_result = subprocess.run([EMULATOR, "--help"],
                                     text=True,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     timeout=15)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("Usage: playback-emulator --schedule=<file>",
                      help_result.stdout)

        decoder = """\
    type: single_error_lut
    block_size: 1
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1]
"""
        self.config.write_text("decoders:\n"
                               "  - id: 0\n" + decoder +
                               "  - id: 1\n" + decoder +
                               "syndrome_mapping:\n"
                               "  - {id: 1, type: final}\n")
        source = self.root / "source.txt"
        source.write_text("1\n0\n")
        self.schedule.write_text("0 stream session=1 source=0 rounds=1\n"
                                 "1 enqueue_data session=1 source=0\n")

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as endpoint:
            endpoint.bind(("127.0.0.1", 0))
            port = endpoint.getsockname()[1]
            result = self.invoke(
                "--backend=inproc",
                f"--udp-endpoint=0:127.0.0.1:{port}",
                f"--source=0:{source}")

        self.assertEqual(result.returncode, 0, result.stderr)
        records = list(csv.DictReader(io.StringIO(result.stdout)))
        self.assertEqual([record["op"] for record in records],
                         ["stream", "enqueue_data"])
        self.assertEqual([record["decoder_id"] for record in records],
                         ["1", "1"])
        self.assertEqual([record["syndrome_bits"] for record in records],
                         ["1", "0"])

    def test_conflicting_endpoints_rejected_before_backend_creation(self):
        result = self.invoke(
            "--backend=cpu_roce",
            "--udp-endpoint=0:127.0.0.1:1234",
            "--cpu-roce-endpoint=0:127.0.0.1:1234",
            "--cpu-roce-timeout=1ms",
            "--cpu-roce-connect-timeout=1ms",
            "--cpu-roce-device=mlx5_0",
            "--cpu-roce-local-ip=127.0.0.1",
            "--cpu-roce-slots=8",
            "--cpu-roce-slot-size=256")
        self.assertEqual(result.returncode, 1)
        self.assertIn("decoder 0 has both a --udp-endpoint= and a "
                      "--cpu-roce-endpoint=", result.stderr)

    def test_cpu_roce_endpoint_requires_device_and_local_ip(self):
        result = self.invoke("--backend=cpu_roce",
                             "--cpu-roce-endpoint=0:127.0.0.1:1234")
        self.assertEqual(result.returncode, 1)
        self.assertIn("--cpu-roce-endpoint= requires --cpu-roce-device= "
                      "and --cpu-roce-local-ip=", result.stderr)

    def test_endpoint_parser_rejects_malformed_values(self):
        cases = (
            ("--udp-endpoint=missing-separator",
             "--udp-endpoint expects ID:HOST:PORT"),
            ("--cpu-roce-endpoint=not-an-id:127.0.0.1:1234",
             "--cpu-roce-endpoint must be a whole number that fits in 64 bits"),
        )
        for argument, expected_error in cases:
            with self.subTest(argument=argument):
                result = self.invoke(argument)
                self.assertEqual(result.returncode, 1)
                self.assertIn(expected_error, result.stderr)

    def test_inproc_static_source_stream_and_final_mapping(self):
        self.config.write_text(self.config.read_text() +
                               "syndrome_mapping:\n"
                               "  - {id: 1, type: final}\n")
        source = self.root / "source.txt"
        source.write_text("1\n0\n")
        self.schedule.write_text("0 stream source=0 rounds=1\n"
                                 "1 enqueue_data source=0\n")

        result = self.invoke("--backend=inproc", f"--source=0:{source}")
        self.assertEqual(result.returncode, 0, result.stderr)
        records = list(csv.DictReader(io.StringIO(result.stdout)))
        self.assertEqual([record["op"] for record in records],
                         ["stream", "enqueue_data"])
        self.assertEqual([record["syndrome_bits"] for record in records],
                         ["1", "0"])

    def test_static_source_file_parser_edges(self):
        self.config.write_text(self.config.read_text() +
                               "syndrome_mapping:\n"
                               "  - {id: 1, type: final}\n")
        self.schedule.write_text("0 stream source=0 rounds=1\n"
                                 "1 enqueue_data source=0\n")
        source = self.root / "source.txt"
        source.write_bytes(b"\r\n1\r\n\n0\r\n")

        result = self.invoke("--backend=inproc", f"--source=0:{source}")
        self.assertEqual(result.returncode, 0, result.stderr)
        records = list(csv.DictReader(io.StringIO(result.stdout)))
        self.assertEqual([record["syndrome_bits"] for record in records],
                         ["1", "0"])

        source.write_bytes(b"1\nx\n")
        result = self.invoke("--backend=inproc", f"--source=0:{source}")
        self.assertEqual(result.returncode, 1)
        self.assertIn(
            f"{source}:2: malformed bit string 'x': expected only '0'/'1'",
            result.stderr)

        missing_source = self.root / "missing-source.txt"
        result = self.invoke("--backend=inproc",
                             f"--source=0:{missing_source}")
        self.assertEqual(result.returncode, 1)
        self.assertIn(f"cannot open source file: {missing_source}",
                      result.stderr)

    def test_csv_stdout(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_reset_csv(result.stdout)

    def test_csv_file(self):
        output = self.root / "result.csv"
        result = self.invoke(f"--out={output}")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assert_reset_csv(output.read_text())

    def test_csv_open_failure(self):
        for output in (self.root / "missing" / "result.csv", self.root):
            with self.subTest(output=output):
                result = self.invoke(f"--out={output}")
                self.assertEqual(result.returncode, 1)
                self.assertIn("cannot open CSV file", result.stderr)
                self.assertIn(str(output), result.stderr)

    @unittest.skipUnless(Path("/dev/full").exists(), "requires /dev/full")
    def test_csv_file_write_failure(self):
        result = self.invoke("--out=/dev/full")
        self.assertEqual(result.returncode, 1)
        self.assertIn("cannot write CSV file: /dev/full", result.stderr)

    @unittest.skipUnless(Path("/dev/full").exists(), "requires /dev/full")
    def test_csv_stdout_write_failure(self):
        with open("/dev/full", "w") as full:
            result = self.invoke(stdout=full)
        self.assertEqual(result.returncode, 1)
        self.assertIn("cannot write CSV to stdout", result.stderr)

    def test_requested_lead_in(self):
        start = time.monotonic()
        result = self.invoke("--lead-in=1s")
        elapsed = time.monotonic() - start
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_reset_csv(result.stdout)
        # Only a lower bound: process startup and a loaded runner may add time.
        self.assertGreaterEqual(elapsed, 1.0)

    def test_zero_lead_in(self):
        result = self.invoke("--lead-in=0ns")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_reset_csv(result.stdout)

    def test_duration_unit_scaling_and_millisecond_overflow(self):
        result = self.invoke("--tick=1us", "--lead-in=1ms",
                             "--udp-timeout=1ms")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_reset_csv(result.stdout)

        result = self.invoke("--udp-timeout=4294967296s")
        self.assertEqual(result.returncode, 1)
        self.assertIn(
            "duration '4294967296s' must be a whole number of milliseconds",
            result.stderr)

    def stim_source(self, **noise):
        return ("--stim-source=0:code=surface_code,task=rotated_memory_z,"
                "distance=3,seed=123," +
                ",".join(f"{key}={value}" for key, value in noise.items()))

    def test_rejects_malformed_and_nonfinite_noise(self):
        for key in NOISE_KEYS:
            for value in ("0.1garbage", "nan", "inf"):
                with self.subTest(key=key, value=value):
                    result = self.invoke(self.stim_source(**{key: value}))
                    self.assertEqual(result.returncode, 1)
                    self.assertIn(f"--stim-source {key}", result.stderr)
                    self.assertIn("finite probability in [0,1]", result.stderr)

    def test_rejects_invalid_noise_numbers(self):
        key = "after_clifford_depolarization"
        for value in ("", "invalid", "1e999", "-0.1", "1.1"):
            with self.subTest(value=value):
                result = self.invoke(self.stim_source(**{key: value}))
                self.assertEqual(result.returncode, 1)
                self.assertIn(f"--stim-source {key}", result.stderr)
                self.assertIn("finite probability in [0,1]", result.stderr)

    def test_valid_noise_probabilities(self):
        self.schedule.write_text("0 stream source=0 rounds=3\n"
                                 "- enqueue_data source=0\n")
        for value in ("0", "1e-3", "+0.1", " 0.1", "0x1p-2"):
            with self.subTest(value=value):
                result = self.invoke(
                    self.stim_source(**{key: value for key in NOISE_KEYS}))
                self.assertEqual(result.returncode, 0, result.stderr)
                records = list(csv.DictReader(io.StringIO(result.stdout)))
                self.assertEqual([r["status"] for r in records], ["100", "100"])
                self.assertEqual([len(r["syndrome_bits"]) for r in records],
                                 [24, 9])

    def test_unit_probability_parses(self):
        result = self.invoke(
            self.stim_source(**{key: "1" for key in NOISE_KEYS}))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_reset_csv(result.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--emulator", required=True)
    args, remaining = parser.parse_known_args()
    EMULATOR = str(Path(args.emulator).resolve())
    unittest.main(argv=[sys.argv[0], *remaining])
