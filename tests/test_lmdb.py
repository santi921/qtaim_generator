import lmdb
import json
import pickle as pkl

from qtaim_gen.source.utils.lmdbs import json_2_lmdbs, inp_files_2_lmdbs


class TestLMDB:
    dir_active = "./test_files/lmdb_tests/"
    chunk_size = 2

    json_2_lmdbs(
        dir_active, dir_active, "charge", "merged_charge.lmdb", chunk_size, clean=True
    )
    json_2_lmdbs(
        dir_active, dir_active, "bond", "merged_bond.lmdb", chunk_size, clean=True
    )
    json_2_lmdbs(
        dir_active, dir_active, "other", "merged_other.lmdb", chunk_size, clean=True
    )
    json_2_lmdbs(
        dir_active, dir_active, "qtaim", "merged_qtaim.lmdb", chunk_size, clean=True
    )

    json_2_lmdbs(
        dir_active,
        dir_active,
        "fuzzy_full",
        "merged_fuzzy.lmdb",
        chunk_size,
        clean=True,
    )

    inp_files_2_lmdbs(
        dir_active, dir_active, "merged_geom.lmdb", chunk_size, clean=True
    )

    charge_lmdb = "./test_files/lmdb_tests/merged_charge.lmdb"
    bond_lmdb = "./test_files/lmdb_tests/merged_bond.lmdb"
    other_lmdb = "./test_files/lmdb_tests/merged_other.lmdb"
    qtaim_lmdb = "./test_files/lmdb_tests/merged_qtaim.lmdb"
    geom_lmdb = "./test_files/lmdb_tests/merged_geom.lmdb"
    fuzzy_lmdb = "./test_files/lmdb_tests/merged_fuzzy.lmdb"

    def read_helper(self, file, lookup):
        env = lmdb.open(
            file,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
        )

        for key, value in env.begin().cursor():
            if str(key.decode("ascii"))[2:-1] == lookup:
                return pkl.loads(value)

    def test_write_read(self):
        orca5_rks_bond = "./test_files/lmdb_tests/orca5_rks/bond.json"
        orca5_qtaim = "./test_files/lmdb_tests/orca5/qtaim.json"
        orca5_uks_charge = "./test_files/lmdb_tests/orca5_uks/charge.json"
        orca5_uks_fuzzy = "./test_files/lmdb_tests/orca5_uks/fuzzy_full.json"
        orca6_rks_other = "./test_files/lmdb_tests/orca6_rks/other.json"

        orca5_rks_bond_json = json.load(open(orca5_rks_bond, "r"))
        orca5_qtaim_json = json.load(open(orca5_qtaim, "r"))
        orca5_uks_charge_json = json.load(open(orca5_uks_charge, "r"))
        orca6_rks_other_json = json.load(open(orca6_rks_other, "r"))
        orca5_rks_fuzzy_json = json.load(open(orca5_uks_fuzzy, "r"))

        dict_orca5 = self.read_helper(self.bond_lmdb, "orca5_rks")
        dict_orca_5 = self.read_helper(self.qtaim_lmdb, "orca5")
        dict_orca5_uks = self.read_helper(self.charge_lmdb, "orca5_uks")
        dict_orca6 = self.read_helper(self.other_lmdb, "orca6_rks")
        dict_orca5_fuzzy = self.read_helper(self.fuzzy_lmdb, "orca5_5ks")

        assert (
            dict_orca5["ibsi"]["3_O_to_10_H"]
            == orca5_rks_bond_json["ibsi"]["3_O_to_10_H"]
        ), f"Expected {orca5_rks_bond_json['ibsi']['3_O_to_10_H']}, got {dict_orca5['ibsi']['3_O_to_10_H']}"
        assert (
            dict_orca_5["1"]["density_alpha"] == orca5_qtaim_json["1"]["density_alpha"]
        ), f"Expected {dict_orca_5['1']['density_alpha']}, got {orca5_qtaim_json['1']['density_alpha']}"
        assert (
            dict_orca5_uks["hirshfeld"]["charge"]["56_H"]
            == orca5_uks_charge_json["hirshfeld"]["charge"]["56_H"]
        ), f"Expected {orca5_uks_charge_json['hirshfeld']['charge']['56_H']}, got {dict_orca5_uks['hirshfeld']['charge']['56_H']}"
        assert (
            dict_orca6["mpp_full"] == orca6_rks_other_json["mpp_full"]
        ), f"Expected {orca6_rks_other_json['mpp_full'] }, got {dict_orca6['mpp_full'] }"

    def test_merge(self):

        for lmdb_file in [
            self.charge_lmdb,
            self.bond_lmdb,
            self.other_lmdb,
            self.qtaim_lmdb,
            self.fuzzy_lmdb,
            self.geom_lmdb,
        ]:
            env = lmdb.open(
                lmdb_file,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
            )

            for key, value in env.begin().cursor():
                if key.decode("ascii") == "length":
                    assert (
                        pkl.loads(value) == 4
                    ), f"Expected 4, got {pkl.loads(value)} on {lmdb_file}"


dir_active = "./test_files/lmdb_tests/"
chunk_size = 2

json_2_lmdbs(
    dir_active,
    dir_active,
    "charge",
    "merged_charge.lmdb",
    chunk_size,
    clean=True,
    merge=False,
)
json_2_lmdbs(
    dir_active,
    dir_active,
    "bond",
    "merged_bond.lmdb",
    chunk_size,
    clean=True,
    merge=False,
)
json_2_lmdbs(
    dir_active,
    dir_active,
    "other",
    "merged_other.lmdb",
    chunk_size,
    clean=True,
    merge=False,
)

json_2_lmdbs(
    dir_active,
    dir_active,
    "fuzzy_full",
    "merged_fuzzy.lmdb",
    chunk_size,
    clean=True,
    merge=False,
)

json_2_lmdbs(
    dir_active,
    dir_active,
    "qtaim",
    "merged_qtaim.lmdb",
    chunk_size,
    clean=True,
    merge=False,
)

inp_files_2_lmdbs(
    dir_active, dir_active, "merged_geom.lmdb", chunk_size, clean=True, merge=False
)

charge_lmdb = "./test_files/lmdb_tests/merged_charge.lmdb"
bond_lmdb = "./test_files/lmdb_tests/merged_bond.lmdb"
other_lmdb = "./test_files/lmdb_tests/merged_other.lmdb"
qtaim_lmdb = "./test_files/lmdb_tests/merged_qtaim.lmdb"
geom_lmdb = "./test_files/lmdb_tests/merged_geom.lmdb"
fuzzy_lmdb = "./test_files/lmdb_tests/merged_fuzzy.lmdb"
obj = TestLMDB()
