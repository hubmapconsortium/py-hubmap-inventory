from setuptools import setup, find_packages

setup(
    name="hubmap-inventory",
    version="2024.01",
    description="Inventory data exploration for HuBMAP datasets",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/hubmapconsortium/hubmap-inventory",
    author="Ivan Cao-Berg, Fatema Shormin, Monica Paz and SAMS Pre-college Data Science Group 2023",
    author_email="icaoberg@psc.edu",
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='HuBMAP, inventory, datasets',
    packages=find_packages(),
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
    python_requires='>=3.6',
    project_urls={
        'Bug Reports': 'https://github.com/hubmapconsortium/hubmap-inventory/issues',
        'Source': 'https://github.com/hubmapconsortium/hubmap-inventory/',
    },
)