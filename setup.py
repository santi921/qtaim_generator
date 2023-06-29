from setuptools import setup, find_packages

# to setup utils run the following
# pip install -e .

setup(
    name="qtaim_gen",
    version="0.0.1",
    packages=find_packages(),
    scripts=[
        "./qtaim_gen/source/scripts/create_files.py",
        "./qtaim_gen/source/scripts/parse_data.py",
        "./qtaim_gen/source/scripts/run.py",
        "./qtaim_gen/source/scripts/helpers/clean_failed_files.py",
        "./qtaim_gen/source/scripts/helpers/check_res_wfn.py",
        "./qtaim_gen/source/scripts/helpers/check_res_rxn_json.py",
    ],
)
