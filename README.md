# graph-litter
A YOLOv5 detector trained on litter database and GNN for explainable learning using GNNExplainer (not yet implemented).

## Installation
Using python 3.9 create virtualenv and activate it:
```sh
python3.9 -m venv venv
source venv/bin/activate
```
Upgrade pip and install requirements:
```sh
pip install --upgrade pip
pip install -r requirements.txt
```
## Run
```sh
python main.py
```
## TODO
- Improve performance of the features extraction (Parallel)
- Put the features in GNN
- Implement GNNExplainer
