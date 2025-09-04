# FastDAG2PAG (minimal)

This repository provides a minimal implementation to convert a given DAG (with a provided set of latent nodes) into a PAG using only oracle D-separation. It focuses strictly on the dag2pag pipeline and omits unrelated local/global causal discovery code.

## Features
- Input: Edge list for a DAG (must be acyclic) and optional set of latent nodes.
- Skeleton learning: Lemma 6.1.3 ancestor-based pruning (single-pass) like `pcalg::dag2pag`.
- Orientation: Currently applies basic unshielded collider orientation (Rule 1). The user requested framework supports Rule 1–4 & 8–10; placeholders can be extended.
- Output: Adjacency matrix (dict-of-dicts) and internal edge mark representation.

## Non-Goals / Exclusions
- No selection bias nodes (explicitly ignored).
- No statistical CI tests (data-free; oracle mode only).
- No full FCI rule set (only collider orientation implemented now; extendable).

## Install (editable local)
```bash
pip install -e .
```

## Quick Example
```python
from fastdag2pag import Dag, dag_to_pag

edges=[('A','B'),('B','C'),('L','A'),('L','C')]  # L is latent
latent={'L'}
dag=Dag(edges)
pag = dag_to_pag(dag, latent)

print('Adjacency:')
for r, row in pag.as_adjacency_matrix().items():
    print(r, row)
print('Edges with marks:', pag.edges_with_marks())
```

## API
`Dag(edges, nodes=None)`  Build a DAG (validates acyclicity).

`dag_to_pag(dag, latent_nodes=None)`  Returns a PAG object.

`PAG.as_adjacency_matrix()`  Returns symmetric 0/1 adjacency for presence of an (any-mark) edge.

## Extending Orientation Rules
`orientation.py` contains a simplified `orient_rules` function. To add Rules 2–4 & 8–10: implement the required pattern searches and mark updates mirroring the richer logic in the original `Learner_Base.orient_rules` method.

## Limitations / TODO
- Implement remaining orientation rules 2–4, 8–10.
- Introduce explicit bidirected (<->) edges when latent confounding is certain.
- Optionally export to common graph formats (Graphviz / networkx MixedGraph view).

## License
Provided as-is for research prototyping; integrate licensing as needed.
# FastDAG2PAG-python