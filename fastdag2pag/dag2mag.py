"""
Main entry for DAG to MAG conversion using D-sep and orientations.
"""
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from fastdag2pag.learner_base import Learner_Base
from fastdag2pag.mixgraph import MixGraph

def dag2mag(dag_df:pd.DataFrame, latent_nodes: list=None, selection_bias_nodes: list=None) -> dict:
    """
    Convert a DAG (adjacency matrix) to MAG using D-sep and orientation rules.
    Args:
        dag_df: pandas.DataFrame, adjacency matrix of DAG
        latent_nodes: list of node names (optional)
        selection_bias: list of node names (optional)
    Returns:
        dict with keys 'MAG.DataFrame', 'MAG.MixGraph', 'DAG.MixGraph'
    """

    learner = Learner_Base(dag_df, ci_type="D_sep", latent_nodes=latent_nodes, selection_bias_nodes=selection_bias_nodes, selection_bias_rules=True)
    skeleton = learner.learn_skeleton_dag2pag()
    mag = learner.orient_by_dag(skeleton)
    dag = MixGraph()
    dag.from_pandas_adjacency(dag_df, graph_type='DAG')
    return {'MAG.DataFrame': mag.MG_to_pandas_adjacency(), 'MAG.MixGraph': mag, 'DAG.MixGraph': dag}


if __name__ == "__main__":
    import numpy as np

    # Create a simple DAG adjacency matrix
    dag_data = np.array([[0, 0, 0],
                         [1, 0, 1],
                         [0, 0, 0]])
    dag_df = pd.DataFrame(dag_data, columns=['A', 'B', 'C'], index=['A', 'B', 'C'])

    result = dag2mag(dag_df, latent_nodes=['B'])
    print("MAG Adjacency Matrix:")
    print(result['MAG.DataFrame'])