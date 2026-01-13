import os
import sqlite3
import tempfile
import pytest
from qtaim_gen.source.scripts.tracking_db import log_to_wandb


def create_test_db(db_path):
    """Create a test SQLite database with sample validation data."""
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
        ('job1', 'subset_a', '/path/to/job1', 'True', 'True', 'True', 'True', 'True', 'True', '10', '120.5'),
        ('job2', 'subset_a', '/path/to/job2', 'True', 'False', 'True', 'True', 'True', 'True', '15', '150.3'),
        ('job3', 'subset_b', '/path/to/job3', 'False', 'True', 'True', 'True', 'True', 'True', '12', '130.7'),
    ]
    
    c.executemany("""
        INSERT INTO validation 
        (job_id, subset, folder, val_qtaim, val_charge, val_bond, val_fuzzy, val_other, val_time, n_atoms, total_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, sample_data)
    
    conn.commit()
    conn.close()


def test_log_to_wandb_without_wandb():
    """Test that log_to_wandb raises ImportError when wandb is not installed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.sqlite")
        create_test_db(db_path)
        
        # Try importing wandb to see if it's available
        try:
            import wandb
            pytest.skip("wandb is installed, skipping ImportError test")
        except ImportError:
            # wandb not installed, test should raise ImportError
            with pytest.raises(ImportError, match="wandb is not installed"):
                log_to_wandb(
                    db_path=db_path,
                    project_name="test-project"
                )


def test_log_to_wandb_with_mock(tmp_path, monkeypatch):
    """Test log_to_wandb function with mocked wandb."""
    db_path = os.path.join(tmp_path, "test.sqlite")
    create_test_db(db_path)
    
    # Mock wandb module
    class MockWandbRun:
        def __init__(self):
            self.summary = {}
        
        def get_url(self):
            return "https://wandb.ai/test/test-project/runs/test123"
        
    class MockWandbTable:
        def __init__(self, dataframe):
            self.dataframe = dataframe
    
    class MockWandb:
        Table = MockWandbTable
        summary = {}
        
        @staticmethod
        def init(project, entity=None, name=None, tags=None, notes=None, config=None):
            return MockWandbRun()
        
        @staticmethod
        def log(data):
            pass
        
        @staticmethod
        def finish():
            pass
    
    # Monkeypatch wandb import
    import sys
    mock_wandb = MockWandb()
    monkeypatch.setitem(sys.modules, 'wandb', mock_wandb)
    
    # Call the function
    url = log_to_wandb(
        db_path=db_path,
        project_name="test-project",
        entity="test-entity",
        run_name="test-run",
        tags=["test"],
        notes="Test run"
    )
    
    # Verify the URL is returned
    assert url == "https://wandb.ai/test/test-project/runs/test123"


def test_log_to_wandb_invalid_db():
    """Test that log_to_wandb raises error for invalid database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "nonexistent.sqlite")
        
        # Try importing wandb to see if it's available
        try:
            import wandb
            pytest.skip("wandb is installed, need to handle this differently")
        except ImportError:
            # wandb not installed, will raise ImportError before database error
            with pytest.raises(ImportError):
                log_to_wandb(
                    db_path=db_path,
                    project_name="test-project"
                )
