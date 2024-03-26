## [CIKM '24] Review-Based Cross-Domain Recommendation via Hyperbolic Embedding and Hierarchy-Aware Domain Disentanglement

### Project Structure

```
.
├── README.md
├── requirements.txt
├── resources
│   ├── Toys and Games.json
│   └── Video Games.json
└── HEAD
    ├── main.py
    ├── model.py
    └── preprocess.py
```

### Setup

- Setup Conda, PyTorch, CUDA
  - > pip install -r requirements.txt
- Download Pretrained Word Embeddings; GloVe (below)
  - https://nlp.stanford.edu/data/glove.6B.zip
  - Place under `'./resources/glove.6B/glove.6B.100d.txt'`
- or Poincare GloVe (below)
  - https://polybox.ethz.ch/index.php/s/TzX6cXGqCX5KvAn/
  - Place under `'./resources/glove.6B/poincare_glove_100D_cosh-dist-sq.txt'`
- Source & Target domain dataset
  - can be excuted with additional downloads: `http://jmcauley.ucsd.edu/data/amazon/`
  - substitute
    - `Toys_and_Games.json`
    - `Video_Games.json`

### Usage

```bash
python3 ./HEAD/main.py  # Training / Test with following code
```

### Experiments

- The NDCG@10 of validation / testing score will be updated at `./results/`
- Iteration will be 300 epochs
- For this version, we remove early stopping to show the overall scores

### Citation

```
N/A
```
