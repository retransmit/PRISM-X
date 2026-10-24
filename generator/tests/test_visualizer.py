"""Automated test suite for Prism Generator Interactive Visualizer and Threat Graph Engine."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from prism_generator.engine import SimulationEngine
from prism_generator.visualizer import (
    compile_simulation_data_bundle,
    generate_visualization_file,
    render_standalone_html,
)


def test_compile_simulation_data_bundle():
    """Verify simulation data bundle structure, entity graphs, and funnel conservation."""
    engine = SimulationEngine(n_cards=30, n_merchants=20, region="US", adversary_mode="intent", seed=42)
    records = engine.generate_batch(n_transactions=300, fraud_prevalence=0.08)

    bundle = compile_simulation_data_bundle(records, region="US", seed=42)

    # 1. Metadata Verification
    assert "metadata" in bundle
    meta = bundle["metadata"]
    assert meta["region"] == "US"
    assert meta["total_transactions"] == 300
    assert meta["fraud_count"] > 0
    assert meta["legitimate_count"] > 0
    assert meta["total_transactions"] == meta["fraud_count"] + meta["legitimate_count"]

    # 2. Macro-Option Breakdown
    assert "macro_options" in bundle
    assert len(bundle["macro_options"]) > 0
    total_macro_count = sum(m["count"] for m in bundle["macro_options"])
    assert total_macro_count == meta["fraud_count"]

    # 3. 4-Hop Funnel Conservation
    assert "switch_funnel" in bundle
    funnel = bundle["switch_funnel"]
    assert funnel["approved"] + funnel["declined"] == 300
    assert len(funnel["hops"]) == 4
    hop_ids = [h["id"] for h in funnel["hops"]]
    assert hop_ids == ["hop_1", "hop_2", "hop_3", "hop_4"]

    # 4. Temporal Series (24 Circadian Buckets)
    assert "temporal_series" in bundle
    assert len(bundle["temporal_series"]) == 24
    assert sum(h["legit"] + h["fraud"] + h["hard_neg"] for h in bundle["temporal_series"]) == 300

    # 5. Threat Graph Topology
    assert "threat_graph" in bundle
    tg = bundle["threat_graph"]
    assert "nodes" in tg
    assert "links" in tg
    assert len(tg["nodes"]) > 0
    node_types = {n["type"] for n in tg["nodes"]}
    assert any(t in node_types for t in ("card", "breach_campaign", "bridge_card"))
    # Links must connect existing nodes
    node_ids = {n["id"] for n in tg["nodes"]}
    for l in tg["links"]:
        assert l["source"] in node_ids
        assert l["target"] in node_ids
        assert l["count"] >= 1

    # 6. Table Records
    assert "table_records" in bundle
    assert len(bundle["table_records"]) > 0
    for r in bundle["table_records"]:
        assert "transaction_id" in r
        assert "card_id" in r
        assert "amount" in r


def test_render_standalone_html():
    """Verify HTML rendering produces valid self-contained HTML with embedded JSON."""
    engine = SimulationEngine(n_cards=20, n_merchants=10, region="IN", adversary_mode="intent", seed=101)
    records = engine.generate_batch(n_transactions=150, fraud_prevalence=0.06)

    bundle = compile_simulation_data_bundle(records, region="IN", seed=101)
    html_str = render_standalone_html(bundle, title="Test Indian Visualizer")

    assert "<!DOCTYPE html>" in html_str
    assert "Prism Generator Visualizer" in html_str
    assert "IN PAYMENT RAILS" in html_str
    assert 'id="threat-svg"' in html_str
    assert 'id="timeline-svg"' in html_str
    assert "const DATA = {" in html_str
    # Must not contain unformatted Python template brackets or errors
    assert "{data_json}" not in html_str


def test_generate_visualization_file_on_disk():
    """Verify end-to-end file generation writes valid HTML to specified path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test_vis.html"
        result_path = generate_visualization_file(
            output_path=str(out_path),
            n_transactions=200,
            region="US",
            fraud_rate=0.05,
            adversary_mode="intent",
            seed=42,
            open_browser=False,
        )

        assert result_path.exists()
        assert result_path.stat().st_size > 10_000
        content = result_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "US PAYMENT RAILS" in content


def test_cli_visualize_command():
    """Verify CLI visualize subcommand executes successfully and produces file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "cli_vis.html"
        cmd = [
            sys.executable,
            "-m",
            "prism_generator.cli",
            "visualize",
            "-n",
            "100",
            "--region",
            "US",
            "-o",
            str(out_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"CLI visualize failed: {res.stderr}"
        assert out_path.exists()
        assert out_path.stat().st_size > 5_000
