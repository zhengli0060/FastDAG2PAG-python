"""
Main entry for DAG to PAG conversion using D-sep and orientation rules.
"""
import pandas as pd
from .learner_base import Learner_Base
from .mixgraph import MixGraph
def dag2pag(dag_df:pd.DataFrame, latent_nodes: list=None, selection_bias: list=None) -> dict:
    """
    Convert a DAG (adjacency matrix) to PAG using D-sep and orientation rules.
    Args:
        dag_df: pandas.DataFrame, adjacency matrix of DAG
        latent_nodes: list of node names (optional)
        selection_bias: list of node names (optional)
    Returns:
        dict with keys 'PAG.DataFrame', 'PAG.MixGraph', 'DAG.MixGraph'
    """
    learner = Learner_Base(dag_df, ci_type="D_sep", latent_nodes=latent_nodes, selection_bias_nodes=selection_bias, selection_bias_rules=True)
    skeleton = learner.learn_skeleton_dag2pag()
    part_directed_graph = learner.orient_collider(skeleton)
    pag = learner.orient_rules(part_directed_graph)
    dag = MixGraph()
    dag.from_pandas_adjacency(dag_df, graph_type='DAG')
    return {'PAG.DataFrame': pag.MG_to_pandas_adjacency(), 'PAG.MixGraph': pag, 'DAG.MixGraph': dag}
