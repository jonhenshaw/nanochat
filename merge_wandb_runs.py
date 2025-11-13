#!/usr/bin/env python3
"""
Merge multiple crashed W&B runs into a single continuous run.

This script downloads history from W&B cloud, deduplicates by step number,
and creates a new merged run with all metrics in proper order.

Usage:
    python merge_wandb_runs.py
"""

import wandb
from wandb import Api
import sys
from collections import defaultdict

def download_run_history(api, entity, project, run_id):
    """Download history from a W&B run."""
    print(f"\n📥 Downloading history for run {run_id}...")

    run = api.run(f"{entity}/{project}/{run_id}")

    # Get all history records
    history = run.scan_history()
    records = []

    for i, record in enumerate(history):
        records.append(record)
        if (i + 1) % 100 == 0:
            print(f"  Downloaded {i + 1} records...", end='\r')

    print(f"  ✓ Downloaded {len(records)} records from {run_id} ({run.state})")
    return records, run

def merge_histories(run_histories, preserve_timestamps=True):
    """
    Merge multiple run histories, deduplicating by step number.

    Args:
        run_histories: List of (records, run) tuples
        preserve_timestamps: If True, keep original timestamps

    Returns:
        List of merged records sorted by step
    """
    print("\n🔀 Merging histories...")

    # Group records by step number
    step_to_record = {}
    step_to_timestamp = {}

    for records, run in run_histories:
        run_id = run.id
        for record in records:
            # Get the actual training step from logged metrics
            if 'step' not in record:
                continue

            step = record['step']
            timestamp = record.get('_timestamp', 0)

            # If we've seen this step before, keep the one with later timestamp
            # (this handles resume overlaps where training re-logs the same step)
            if step in step_to_record:
                if timestamp > step_to_timestamp[step]:
                    step_to_record[step] = record
                    step_to_timestamp[step] = timestamp
                    print(f"  ⚠ Step {step} duplicated, keeping later version", end='\r')
            else:
                step_to_record[step] = record
                step_to_timestamp[step] = timestamp

    # Sort by step number
    sorted_steps = sorted(step_to_record.keys())
    merged_records = [step_to_record[step] for step in sorted_steps]

    print(f"\n  ✓ Merged into {len(merged_records)} unique steps")
    print(f"  Step range: {sorted_steps[0]} → {sorted_steps[-1]}")

    # Count duplicates
    total_records = sum(len(records) for records, _ in run_histories)
    duplicates = total_records - len(merged_records)
    if duplicates > 0:
        print(f"  Removed {duplicates} duplicate step records")

    return merged_records

def create_merged_run(entity, project, merged_records, original_runs):
    """Create a new W&B run with merged history."""
    print("\n📤 Creating merged run...")

    # Get config from the most recent run
    latest_run = original_runs[-1]
    config = dict(latest_run.config)

    # Add merge metadata
    config['_merged'] = True
    config['_merged_from_runs'] = [run.id for run in original_runs]
    config['_merged_run_count'] = len(original_runs)

    # Initialize new run
    merged_run_name = f"{latest_run.name}-merged"
    run = wandb.init(
        project=project,
        entity=entity,
        name=merged_run_name,
        config=config,
        reinit=True
    )

    print(f"  Creating run: {merged_run_name}")

    # Log all merged history
    print(f"  Logging {len(merged_records)} records...")
    for i, record in enumerate(merged_records):
        # Remove W&B internal fields that shouldn't be re-logged
        clean_record = {k: v for k, v in record.items()
                       if not k.startswith('_') or k == '_timestamp'}

        run.log(clean_record)

        if (i + 1) % 100 == 0:
            print(f"  Logged {i + 1}/{len(merged_records)} records...", end='\r')

    print(f"\n  ✓ Logged all {len(merged_records)} records")

    # Finish the run
    run.finish()

    print(f"\n✅ Merge complete!")
    print(f"\n🔗 View merged run: {run.url}")

    return run

def main():
    # Configuration
    ENTITY = "djhenny"
    PROJECT = "nanochat"
    RUN_IDS = [
        "url355b1",   # Run 1: crashed Nov 11, 03:59
        "h9whxpg8",   # Run 2: crashed Nov 11, 20:18
        "jla51cre",   # Run 3: running Nov 12, 17:20
    ]

    print("=" * 60)
    print("W&B Run Merger")
    print("=" * 60)
    print(f"\nEntity: {ENTITY}")
    print(f"Project: {PROJECT}")
    print(f"Runs to merge: {len(RUN_IDS)}")
    for run_id in RUN_IDS:
        print(f"  - {run_id}")

    # Initialize API
    api = Api()

    # Download all run histories
    run_histories = []
    for run_id in RUN_IDS:
        try:
            records, run = download_run_history(api, ENTITY, PROJECT, run_id)
            run_histories.append((records, run))
        except Exception as e:
            print(f"  ✗ Error downloading {run_id}: {e}")
            sys.exit(1)

    # Merge histories
    merged_records = merge_histories(run_histories, preserve_timestamps=True)

    if not merged_records:
        print("\n✗ No records to merge!")
        sys.exit(1)

    # Create merged run
    original_runs = [run for _, run in run_histories]
    merged_run = create_merged_run(ENTITY, PROJECT, merged_records, original_runs)

    print("\n" + "=" * 60)
    print("Done! 🎉")
    print("=" * 60)

if __name__ == "__main__":
    main()
