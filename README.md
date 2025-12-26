# 🏍️ FastDAG2PAG-python

**FastDAG2PAG-python** is a Python tool for converting **Directed Acyclic Graphs (DAGs)** into **Partial Ancestral Graphs (PAGs)**.  
It is inspired by the `dag2pag` function from the R package [pcalg](https://cran.r-project.org/web/packages/pcalg/index.html) 🧠, reimplemented in Python for improved accessibility and integration into modern workflows.

## ⚠️ Notes

- The skeleton learning implemented in this tool follows the same theoretical foundation as the `dag2pag` in the R package **pcalg**.  
  - Reference: Richardson, T. S., & Spirtes, P. (2002). Ancestral Graph Markov Models, Theorem 4.2.
- The core logic has been **refactored** and **optimized** for Python.

- 2025/12/26: I added a new function 'dag2mag' to convert DAGs to MAGs.
  
**Additionally, I have discovered a potential issue in pcalg's dag2pag implementation. For details, please see `compare_pcalg.ipynb` in this repository.**

## 🚀 Usage Example
See `example_dag2pag.ipynb`.


## 🛠️ Dependencies
- networkx==3.2.1
- scipy==1.11.4
- igraph==0.11.8
- pgmpy==0.1.19
- graphviz==0.20.3
- pydot==3.0.2
- causallearn==the newest version(2025/11/20)(for DAGtoPAGPython part only)


## 🤝 Contributing & Contact

If you have questions or suggestions, feel free to open an issue or contact me directly.

Email: [zhengli0060(at)gmail(dot)com]
