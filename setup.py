from setuptools import setup

setup(
    name='hubmapbaginventory',
    version='2023.03',
    description='Generates inventory files for HuBMAP datasets',
    url='https://github.com/hubmapconsortium/hubmap-inventory',
    author='Ivan Cao-Berg',
    author_email='icaoberg@psc.edu',
    packages=['hubmapbags'],
    install_requires=['pandas',
        'numpy',
        'tabulate',
        'pandarallel',
        'tqdm'],
)



