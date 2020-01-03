# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     notebook_metadata_filter: all,-language_info,-toc,-latex_envs
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Initial setup
#
#
# 1. Install miniconda for you architecture:  https://docs.conda.io/en/latest/miniconda.html
#
# ## For MacOS
#
# a. Start a terminal and type:
#
# ```
# conda install git
# ```
#
# b. cd to your home directory and make a folder called repos
#
# ```
# mkdir repos
# cd repos
# ```
#
# c. clone the course repository and cd into it
#
# ```
# git clone https://github.com/eldadHaber/ComputationalGeoCourse
# cd ComputationalGeoCourse
# ```
#
# d. checkout the pha branch
#
# ```
# git checkout -b pha origin/pha
# ```
#
# e. cd to the conda folder and create and activate the e213 environment
#
# ```
# cd conda
# conda env create -f environment.yml
# conda activate e213
# ```
#
# f. change back to the notebooks folder and start jupyter
#
# ```
# cd ../notebooks
# jupyter notebook
# ```
#
#
