# /// script
# requires-python = ">=3.10"
# dependencies = ["loguru"]
# ///
"""Convert RelBench rel-stack data_out.json into standardized exp_sel_data_out.json format.

Each FK link with both child and parent features becomes a separate "dataset" entry.
Each aligned row becomes one example: input=parent features, output=child features.
"""

import json
import sys
import hashlib
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

WORKSPACE = Path(__file__).parent
INPUT_PATH = WORKSPACE / "temp" / "datasets" / "data_out.json"
OUTPUT_PATH = WORKSPACE / "full_data_out.json"

# Limit rows per FK link to keep output manageable (~30MB total)
MAX_EXAMPLES_PER_LINK = 5_000


def make_fold(row_idx: int, n_folds: int = 5) -> int:
    """Deterministic fold assignment based on row index hash."""
    h = int(hashlib.md5(str(row_idx).encode()).hexdigest(), 16)
    return h % n_folds


def main():
    logger.info(f"Loading data from {INPUT_PATH}")
    data = json.loads(INPUT_PATH.read_text())

    dataset_name = data["dataset_name"]  # "rel-stack"
    logger.info(f"Dataset: {dataset_name}, FK links: {len(data['fk_links'])}")

    datasets = []

    for fk in data["fk_links"]:
        child_feats = fk.get("child_feature_cols", [])
        parent_feats = fk.get("parent_feature_cols", [])
        child_data = fk.get("child_features", [])
        parent_data = fk.get("parent_features", [])
        link_id = fk["link_id"]

        # Skip FK links without both child and parent features
        if not child_feats or not parent_feats:
            logger.info(f"  Skipping {link_id}: child_feats={len(child_feats)}, parent_feats={len(parent_feats)}")
            continue

        # Skip if no aligned data
        if not child_data or not parent_data:
            logger.info(f"  Skipping {link_id}: no aligned data")
            continue

        n_rows = min(len(child_data), len(parent_data), MAX_EXAMPLES_PER_LINK)
        logger.info(f"  Processing {link_id}: {n_rows} rows, "
                    f"child_feats={child_feats}, parent_feats={parent_feats}")

        examples = []
        for i in range(n_rows):
            # Build input: parent features as JSON dict string
            parent_dict = {}
            for j, col_name in enumerate(parent_feats):
                parent_dict[col_name] = parent_data[i][j] if j < len(parent_data[i]) else None
            input_str = json.dumps(parent_dict)

            # Build output: child features as JSON dict string
            child_dict = {}
            for j, col_name in enumerate(child_feats):
                child_dict[col_name] = child_data[i][j] if j < len(child_data[i]) else None
            output_str = json.dumps(child_dict)

            example = {
                "input": input_str,
                "output": output_str,
                "metadata_fold": make_fold(i),
                "metadata_row_index": i,
                "metadata_task_type": "cross_table_prediction",
                "metadata_child_table": fk["child_table"],
                "metadata_parent_table": fk["parent_table"],
                "metadata_fkey_col": fk["fkey_col"],
                "metadata_child_feature_names": child_feats,
                "metadata_parent_feature_names": parent_feats,
                "metadata_cardinality_mean": fk.get("cardinality_mean", 0.0),
                "metadata_cardinality_median": fk.get("cardinality_median", 0.0),
                "metadata_coverage": fk.get("coverage", 0.0),
            }
            examples.append(example)

        ds_name = f"rel-stack/{link_id}"
        datasets.append({
            "dataset": ds_name,
            "examples": examples,
        })
        logger.info(f"    -> {ds_name}: {len(examples)} examples")

    # Assemble final output
    output = {
        "metadata": {
            "source": data.get("source", "RelBench (Stanford SNAP)"),
            "domain": data.get("domain", "social/Q&A platform"),
            "description": data.get("description", ""),
            "license": data.get("license", "CC BY-SA 4.0"),
            "citation": data.get("citation", ""),
            "num_tables": len(data.get("tables", {})),
            "num_fk_links": len(data.get("fk_links", [])),
            "num_tasks": len(data.get("tasks", [])),
            "task_names": [t["task_name"] for t in data.get("tasks", [])],
            "table_names": list(data.get("tables", {}).keys()),
            "schema_topology": data.get("schema_topology", {}),
        },
        "datasets": datasets,
    }

    total_examples = sum(len(ds["examples"]) for ds in datasets)
    logger.info(f"Total: {len(datasets)} datasets, {total_examples} examples")

    logger.info(f"Writing output to {OUTPUT_PATH}")
    out_text = json.dumps(output, indent=2)
    OUTPUT_PATH.write_text(out_text)
    logger.info(f"Output: {len(out_text) / 1e6:.1f} MB, {len(out_text)} chars")
    logger.info("Done!")


if __name__ == "__main__":
    main()
