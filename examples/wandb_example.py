#!/usr/bin/env python
"""
Example script demonstrating wandb integration with tracking_db.py

This script shows how to use the log_to_wandb function to upload
SQLite tracking data to Weights & Biases.
"""

import os
import sqlite3
import tempfile
from qtaim_gen.source.scripts.tracking_db import log_to_wandb


def create_example_db(db_path):
    """Create an example SQLite database with sample validation data."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create a simple validation table
    c.execute("""
        CREATE TABLE validation (
            job_id TEXT,
            subset TEXT,
            folder TEXT,
            val_qtaim TEXT,
            val_charge TEXT,
            val_bond TEXT,
            val_fuzzy TEXT,
            val_other TEXT,
            val_time TEXT,
            n_atoms TEXT,
            total_time TEXT,
            PRIMARY KEY(job_id, subset, folder)
        )
    """)
    
    # Insert sample data
    sample_data = [
        ('mol_001', 'organic', '/jobs/organic/mol_001', 'True', 'True', 'True', 'True', 'True', 'True', '25', '342.1'),
        ('mol_002', 'organic', '/jobs/organic/mol_002', 'True', 'True', 'True', 'True', 'True', 'True', '18', '201.5'),
        ('mol_003', 'inorganic', '/jobs/inorganic/mol_003', 'True', 'False', 'True', 'True', 'True', 'True', '42', '567.3'),
        ('mol_004', 'inorganic', '/jobs/inorganic/mol_004', 'False', 'True', 'True', 'True', 'True', 'True', '31', '421.8'),
        ('mol_005', 'organic', '/jobs/organic/mol_005', 'True', 'True', 'False', 'True', 'True', 'True', '22', '289.6'),
    ]
    
    c.executemany("""
        INSERT INTO validation 
        (job_id, subset, folder, val_qtaim, val_charge, val_bond, val_fuzzy, val_other, val_time, n_atoms, total_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, sample_data)
    
    conn.commit()
    conn.close()
    print(f"Created example database at: {db_path}")


if __name__ == "__main__":
    # Create a temporary database for demonstration
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sqlite') as f:
        db_path = f.name
    
    # Create example data
    create_example_db(db_path)
    
    print("\nExample database created with sample validation data.")
    print("\nTo log this data to W&B, you would run:")
    print(f"""
    from qtaim_gen.source.scripts.tracking_db import log_to_wandb
    
    log_to_wandb(
        db_path="{db_path}",
        project_name="qtaim-tracking-example",
        entity="your-username-or-team",
        run_name="example-run",
        tags=["example", "demo"],
        notes="Example tracking database upload"
    )
    """)
    
    print("\nNote: You need to install wandb and authenticate first:")
    print("  pip install wandb")
    print("  wandb login")
    
    # Clean up
    print(f"\nTemporary database created at: {db_path}")
    print("(Delete this file when done)")
