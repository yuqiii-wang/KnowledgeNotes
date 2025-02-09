# https://labelstud.io/guide/install#Install-using-pip

python3 -m venv env
source env/bin/activate
python -m pip install label-studio

label-studio

#### Some manual labelling work
# https://labelstud.io/guide/export#spaCy
# https://spacy.io/usage/training#data-convert
python -m spacy convert /path/to/<filename>.conll -c conll .

python -m spacy init fill-config base_config.cfg config.cfg

python -m spacy train config.cfg --output ./output --code custom_functions.py