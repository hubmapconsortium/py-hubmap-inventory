from setuptools import setup

setup(
    name="hubmap-inventory",
    version="2023.03",
    description="Inventory data exploration for HuBMAP datasets",
    url="https://github.com/hubmapconsortium/hubmap-inventory",
    author="Ivan Cao-Berg, Fatema Shormin, Monica Paz and SAMS Pre-college Data Science Group",
    author_email="icaoberg@psc.edu",
    install_requires=[
        "pandas",
        "numpy",
        "tabulate",
        "pandarallel",
        "tqdm",
        "numpyencoder",
        "python-magic",
        "humanize",
    ],
    packages=["hubmapinventory"],
)
