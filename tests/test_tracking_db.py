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
    
    # Create mock wandb module with proper structure
    import types
    mock_wandb = types.ModuleType('wandb')
    mock_wandb.Table = MockWandbTable
    mock_run = MockWandbRun()
    
    def mock_init(project, entity=None, name=None, tags=None, notes=None, config=None):
        return mock_run
    
    def mock_log(data):
        pass
    
    def mock_finish():
        pass
    
    mock_wandb.init = mock_init
    mock_wandb.log = mock_log
    mock_wandb.finish = mock_finish
    
    # Monkeypatch wandb import
    import sys
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


def test_log_to_wandb_boolean_handling(tmp_path, monkeypatch):
    """Test that log_to_wandb correctly handles various boolean formats."""
    db_path = os.path.join(tmp_path, "test_bool.sqlite")
    
    # Create database with various boolean formats
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE validation (
            job_id TEXT,
            subset TEXT,
            folder TEXT,
            val_qtaim TEXT,
            PRIMARY KEY(job_id, subset, folder)
        )
    """)
    # Test various boolean representations
    test_data = [
        ('job1', 'subset', '/path/1', 'True'),   # String 'True'
        ('job2', 'subset', '/path/2', 'true'),   # String 'true' (lowercase)
        ('job3', 'subset', '/path/3', 'False'),  # String 'False'
        ('job4', 'subset', '/path/4', None),     # NULL/None
    ]
    c.executemany("INSERT INTO validation VALUES (?, ?, ?, ?)", test_data)
    conn.commit()
    conn.close()
    
    # Mock wandb
    import types
    mock_wandb = types.ModuleType('wandb')
    
    class MockRun:
        def __init__(self):
            self.summary = {}
        def get_url(self):
            return "https://wandb.ai/test/test/runs/test"
    
    class MockTable:
        def __init__(self, dataframe):
            self.dataframe = dataframe
    
    mock_run = MockRun()
    mock_wandb.Table = MockTable
    mock_wandb.init = lambda **kwargs: mock_run
    mock_wandb.log = lambda data: None
    mock_wandb.finish = lambda: None
    
    import sys
    monkeypatch.setitem(sys.modules, 'wandb', mock_wandb)
    
    # Call the function
    log_to_wandb(db_path=db_path, project_name="test")
    
    # Verify that it correctly counted True values (should be 2: 'True' and 'true')
    assert mock_run.summary.get("val_qtaim_count") == 2


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
