#!/bin/bash -v
#python $sphinxlib/check_rendered.py notebooks
rsync -avz ../notebooks/* doc_notebooks/
sphinx-build -N -v -b html . _build
