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
    license='GPLv3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='HuBMAP, inventory, datasets',
    packages=find_packages(),
    install_requires=[
        "matplotlib==3.7.2",
        "numpy==1.25.2",
        "pandarallel==1.6.5",
        "pandas==2.0.3",
        "setuptools==67.8.0",
        "tabulate==0.9.0",
        "tqdm==4.65.0",
        "numpyencoder==0.3.0",
        "python-magic==0.4.27",
        "humanize==4.9.0",
    ],
    python_requires='>=3.6',
    project_urls={
        'Bug Reports': 'https://github.com/hubmapconsortium/hubmap-inventory/issues',
        'Source': 'https://github.com/hubmapconsortium/hubmap-inventory/',
    },
)
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a dataframe named 'df' with 'group_name' and 'status' columns
# and you have already imported the required libraries.

# Pivot the DataFrame to get the 'status' as columns and 'group_name' as index
pivot_df = df.pivot_table(index='group_name', columns='status', aggfunc='size', fill_value=0)

# Create a bar plot for each 'status' representing each 'group_name'
ax = pivot_df.plot(kind='bar', stacked=True, figsize=(12, 6))

# Customize the plot
plt.title("Status Distribution for Each Group")
plt.xlabel("Group Name")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()
