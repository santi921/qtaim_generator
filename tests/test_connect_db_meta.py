import os
import lmdb
import pickle
import tempfile
from qtaim_gen.source.core.converter import Converter
from tests.utils_lmdb import _make_lmdb




def test_connect_db_with_meta_uses_length_entry(tmp_path):
    lmdb_file = os.path.join(tmp_path, "meta_test.lmdb")
    _make_lmdb(lmdb_file, with_length=True)

    result = Converter.connect_db(None, lmdb_path=lmdb_file, with_meta=True)
    assert isinstance(result, dict)
    assert "env" in result and "keys" in result and "num_samples" in result
    # length entry should be read (2)
    assert result["num_samples"] == 2
    # keys should include the two data keys
    assert len(result["keys"]) == 2


def test_connect_db_with_meta_no_length_counts_keys(tmp_path):
    lmdb_file = os.path.join(tmp_path, "meta_test_nolength.lmdb")
    _make_lmdb(lmdb_file, with_length=False)

    result = Converter.connect_db(None, lmdb_path=lmdb_file, with_meta=True)
    assert isinstance(result, dict)
    # when no length entry exists, num_samples should equal discovered key count
    assert result["num_samples"] == len(result["keys"]) == 2
