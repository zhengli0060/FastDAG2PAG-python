# 🏍️ FastDAG2PAG-python

**FastDAG2PAG-python** is a Python tool for converting **Directed Acyclic Graphs (DAGs)** into **Partial Ancestral Graphs (PAGs)**.  
It is inspired by the `dag2pag` function from the R package [pcalg](https://cran.r-project.org/web/packages/pcalg/index.html) 🧠, reimplemented in Python for improved accessibility and integration into modern workflows.

## ⚠️ Notes

- The skeleton learning implemented in this tool **does not** account for selection bias, following the same theoretical foundation as the `dag2pag` in the R package **pcalg**.  
  - Reference: Richardson, T. S., & Spirtes, P. (2002). Ancestral Graph Markov Models, Theorem 4.2.
- The core logic has been **refactored** and **optimized** for Python.

**Additionally, I have discovered a potential issue in pcalg's dag2pag implementation. For details, please see `compare_pcalg.ipynb` in this repository.**

## 🚀 Usage Example
See `example_dag2pag.ipynb`.

## 🤝 Contributing & Contact

If you have questions or suggestions, feel free to open an issue or contact me directly.

Email: [zhengli0060(at)gmail(dot)com]
