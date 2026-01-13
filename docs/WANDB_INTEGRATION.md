# W&B Integration for tracking_db.py

This document describes the Weights & Biases (wandb) integration for the QTAIM tracking database.

## Overview

The `log_to_wandb` function enables you to upload your SQLite tracking database to Weights & Biases for online monitoring, visualization, and collaboration.

## Installation

Install wandb as an optional dependency:

```bash
pip install qtaim_generator[wandb]
```

Or install wandb directly:

```bash
pip install wandb
```

## Authentication

Before using wandb, you need to authenticate:

```bash
wandb login
```

This will prompt you for your API key, which you can find at https://wandb.ai/authorize

## Usage

### Basic Usage

```python
from qtaim_gen.source.scripts.tracking_db import log_to_wandb

# Log your tracking database to W&B
log_to_wandb(
    db_path="validation_results.sqlite",
    project_name="qtaim-tracking"
)
```

### Full Example with All Options

```python
from qtaim_gen.source.scripts.tracking_db import log_to_wandb

url = log_to_wandb(
    db_path="validation_results.sqlite",
    project_name="qtaim-tracking",
    entity="my-team",  # Your W&B username or team name
    run_name="experiment-2024-01-13",  # Name for this run
    tags=["production", "full-dataset", "wave-2"],  # Tags for organization
    notes="QTAIM validation results for OMol dataset",  # Description
    config={  # Optional configuration to log
        "dataset": "OMol-4M",
        "method": "Multiwfn",
        "level_of_theory": "B3LYP/def2-TZVP"
    },
    table_name="validation_tracking"  # Name for the W&B table
)

print(f"View results at: {url}")
```

## What Gets Logged

The `log_to_wandb` function logs:

1. **Main Table**: The entire `validation` table from your SQLite database as a W&B Table
2. **Summary Statistics**:
   - Total number of jobs
   - Number of unique subsets
   - Count of jobs passing each validation check (val_qtaim, val_charge, val_bond, val_fuzzy, val_other, val_time)

## Integration with Existing Workflow

You can integrate wandb logging into your existing tracking workflow:

```python
from qtaim_gen.source.scripts.tracking_db import scan_and_store_parallel, log_to_wandb

# First, scan and store results in SQLite
scan_and_store_parallel(
    root_dir="/path/to/jobs",
    db_path="validation_results.sqlite",
    max_workers=8
)

# Then, upload to W&B for online tracking
log_to_wandb(
    db_path="validation_results.sqlite",
    project_name="qtaim-tracking",
    run_name="scan-2024-01-13"
)
```

## Benefits

- **Online Access**: View your tracking data from anywhere
- **Visualization**: Use W&B's built-in visualization tools
- **Collaboration**: Share results with team members
- **History**: Keep a history of all your tracking runs
- **Filtering**: Use W&B's table filtering and sorting capabilities

## Parameters

- `db_path` (str, required): Path to your SQLite database
- `project_name` (str, required): Name of the W&B project
- `entity` (str, optional): W&B username or team name
- `run_name` (str, optional): Name for this run (W&B generates one if not provided)
- `tags` (list, optional): List of tags for organization
- `notes` (str, optional): Description or notes for this run
- `config` (dict, optional): Configuration dictionary to log
- `table_name` (str, optional): Name for the W&B table (default: "validation_tracking")

## Return Value

Returns the URL to the W&B run where you can view your data.

## Error Handling

The function will raise:
- `ImportError` if wandb is not installed
- `sqlite3.Error` if there's an error reading the database

## Example Output

```
Data logged to W&B: https://wandb.ai/your-team/qtaim-tracking/runs/abc123xyz
```

## See Also

- [W&B Documentation](https://docs.wandb.ai/)
- [W&B Tables Guide](https://docs.wandb.ai/guides/tables)
