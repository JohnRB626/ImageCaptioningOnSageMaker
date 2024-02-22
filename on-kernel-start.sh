# This script installs a single pip module on a SageMaker Studio Kernel Application
#!/bin/bash

set -eux

python -m pip install -U spacy
python -m spacy download en_core_web_md