from setuptools import setup

setup(
    name="hubmap-inventory",
    version="2023.03",
    description="Generates inventory files for HuBMAP datasets",
    url="https://github.com/hubmapconsortium/hubmap-inventory",
    author="Ivan Cao-Berg",
    author_email="icaoberg@psc.edu",
    install_requires=[
        "pandas",
        "numpy",
        "tabulate",
        "pandarallel",
        "tqdm",
        "numpyencoder",
        "python-magic",
    ],
    packages=["hubmapinventory"],
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
