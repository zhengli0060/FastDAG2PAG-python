import logging
import time
from typing import List, Optional, Union
from .mixgraph import MixGraph
from .sepset import Separation_Set
import pandas as pd
import numpy as np
from .Graph_utils import Edge, Mark, Node
from .ci_test import CI_test
from itertools import combinations
import networkx as nx

logger = logging.getLogger("Learner_Base") 
logger.setLevel(logging.INFO)


class Learner_Base:

    def __init__(self, data: pd.DataFrame, **kwargs):

        self.latent_nodes = kwargs.get("latent_nodes", None)  # List of latent nodes, default is None
        self.selection_bias_nodes = kwargs.get("selection_bias_nodes", None)

        if self.latent_nodes is not None and self.selection_bias_nodes is not None:
            assert not (set(self.latent_nodes) & set(self.selection_bias_nodes)), "latent_nodes and selection_bias_nodes have overlapping elements"


        self.DAG = nx.DiGraph(data)  # Create a directed graph from the adjacency matrix, for dag to pag
            
        self.ci_test = CI_test(data, method_type='D_sep', selection_bias_nodes=self.selection_bias_nodes)

        observed_data = data.drop(columns=self.latent_nodes) if self.latent_nodes is not None else data
        observed_data = observed_data.drop(columns=self.selection_bias_nodes) if self.selection_bias_nodes is not None else observed_data
        self._init_nodes(observed_data)
        self.pag = MixGraph(incoming_graph_data=self.Nodes_list) # Nodes_list only includes observed nodes, so pag is a MixGraph with only observed nodes.
        self.sepsets = Separation_Set(set(self.Nodes_list))
  

        self.selection_bias_rules = kwargs.get("selection_bias_rules", False)  # Whether to consider selection bias in orient rules, default is False
        if self.selection_bias_nodes is not None and not self.selection_bias_rules:
            raise Warning("selection_bias_nodes is set but selection_bias_rules is False. No selection bias rules will be applied.")
        
    def get_ci_test_number(self) -> int:
        """
        Get the number of CI tests performed.
        """
        return self.ci_test.get_ci_num()  # including the number of CI tests performed in the MB learning process and skeleton learning process
    
    def _init_nodes(self, data: pd.DataFrame):

        if isinstance(data, pd.DataFrame):
            self.Nodes_list = [Node(node_name, index) for index, node_name in enumerate(data.columns)]
            self.vars_list = data.columns.tolist()
            self.Nodes_dict = {node.name: node for node in self.Nodes_list}
        else:
            raise TypeError("Data must be a pandas DataFrame.")

 
    def learn_skeleton_dag2pag(self) -> MixGraph:

        """
        The skeleton learning function for dag to pag.
        We here reference Theorem 4.2 in Richardson et al. to learn the skeleton fastly, same with the function 'dag2pag' in R package 'pcalg'.
        - Richardson, T. S., & Spirtes, P. (2002). Ancestral Graph Markov Models.
        """


        graph = MixGraph(incoming_graph_data=self.Nodes_list)
        graph._init_complete_graph()
        ancList = {}
        
        for var in self.vars_list:
            ancList[var] = set(nx.ancestors(self.DAG, var))

        if self.selection_bias_nodes is not None:
            for sel_var in self.selection_bias_nodes:
                ancList[sel_var] = set(nx.ancestors(self.DAG, sel_var))



        for x, y in combinations(self.vars_list, 2):
            sepset = ancList[x] | ancList[y]
            if self.selection_bias_nodes is not None:
                for sel_var in self.selection_bias_nodes:
                    sepset = sepset | ancList[sel_var]

            sepset = sepset - {x, y} 
            if self.latent_nodes is not None:
                sepset = sepset - set(self.latent_nodes)

            # we have set the conditioning set including selection bias nodes in ci_test 
            if self.selection_bias_nodes is not None:
                sepset = sepset - set(self.selection_bias_nodes)
            
            if self.ci_test(x, y, list(sepset))[0]:
                self.sepsets._add(self.Nodes_dict[x], self.Nodes_dict[y], set(self.Nodes_dict[name] for name in sepset)) # the condition set = sepset \cup selection_bias_nodes
                graph.remove_Edge(self.Nodes_dict[x], self.Nodes_dict[y]) # 
                logger.info(f'remove {x} -- {y} via has sepset')

        return graph


    def orient_collider(self, undirected_graph: MixGraph) -> MixGraph:

        Cand_triplets = undirected_graph.find_unique_triplets()
        for (z, y, x) in Cand_triplets:
            if self.sepsets.has_sepset(x, z): # in local causal discovery, we check the existence of sepset of x and z
                if not self.sepsets.is_in_sepset(target=y, node1=x, node2=z):
                    undirected_graph.update_Edge(node1=x, lmark=None, rmark=Mark.ARROW, node2=y)
                    undirected_graph.update_Edge(node1=z, lmark=None, rmark=Mark.ARROW, node2=y)
                    logger.info(f"Orienting collider: {x} *-> {y} <-* {z}")

        return undirected_graph
    
    def orient_rules(self, pag: MixGraph) -> MixGraph:

        """
        Apply orientation rules to the PAG.
        Reference: Zhang, J. (2008). On the completeness of orientation rules for causal discovery in the presence of latent confounders and selection bias, Artificial Intelligence 172 (16-17), 1873-1896.

        The logical structure is inspired by the pcalg implementation, but all rules are re-implemented here.
        """
        
        update_flag = True
        while update_flag:  # Continue until no changes are made
            update_flag = False
            pag, update_flag = self.Rule_1(pag, update_flag)
            pag, update_flag = self.Rule_2(pag, update_flag)
            pag, update_flag = self.Rule_3(pag, update_flag)
            pag, update_flag = self.Rule_4(pag, update_flag)

        if self.selection_bias_rules:
            update_flag = True
            while update_flag:  # Continue until no changes are made
                update_flag = False
                pag,update_flag = self.Rule_5(pag, update_flag)  
                pag, update_flag = self.Rule_6(pag, update_flag)  
                pag, update_flag = self.Rule_7(pag, update_flag)  

        update_flag = True
        while update_flag:  # Continue until no changes are made
            update_flag = False
            pag, update_flag = self.Rule_8(pag, update_flag)
            pag, update_flag = self.Rule_9(pag, update_flag)
            pag, update_flag = self.Rule_10(pag, update_flag)

        return pag


    def Rule_1(self, pag: MixGraph, update_flag: bool) -> MixGraph:
        """
        If a *-> b o-* r, and a and r are not adjacent, then orient the triple as a *-> b -> r.
        """
        for b, r in pag.get_circ_star_Edge(): # Get edges of the form b o-* r
            for a in pag.get_into_nodes(b):
                if self.sepsets.has_sepset(a, r) and \
                    self.sepsets.is_in_sepset(target=b, node1=a, node2=r):
                        pag.update_Edge(node1=b, lmark=Mark.TAIL, rmark=Mark.ARROW, node2=r)
                        update_flag = True
                        logger.info(f"Orienting Rule 1: {b} o-* {r} to {b} --> {r}")
                        break # No need to check other As for this b, r pair

        return pag, update_flag        

    def Rule_2(self, pag: MixGraph, update_flag: bool) -> MixGraph:
        """
        If a -> b *-> r or a *-> b -> r, and a *–○ r, then orient a *–○ r as a *-> r.
        """
        for r, a in pag.get_circ_star_Edge():  # edges of the form a *-o r
            for b in pag.get_into_nodes(r):  # b *-> r
                if pag.has_directed_Edge(a, b) or \
                    (pag.has_into_Edge(a, b) and pag.has_out_Edge(b, r)):
                    pag.update_Edge(node1=a, lmark=None, rmark=Mark.ARROW, node2=r)
                    update_flag = True
                    logger.info(f"Orienting Rule 2: {a} *-o {r} to {a} *-> {r}")
                    break
        return pag, update_flag

    def Rule_3(self, pag: MixGraph, update_flag: bool) -> MixGraph:
        """
        If a *-> b <-* r, a *-o t o-* r, a and r are not adjacent, and t *-o b,
        then orient t *-o b as t *-> b.
        """
        for b, t in pag.get_circ_star_Edge():  # Get edges of the form t *-o b
            Cand_ar = pag.get_into_nodes(b)   # a or r *-> b
            Cand_ar = {a for a in Cand_ar if pag.has_into_Edge(a, b)}  # Only consider a *-> b
            if len(Cand_ar) >= 2:
                for a, r in combinations(Cand_ar, 2):
                    if self.sepsets.has_sepset(a, r) and \
                        self.sepsets.is_in_sepset(target=t, node1=a, node2=r):  # otherwise t is a collider on <a, t, r>
                            pag.update_Edge(node1=t, lmark=None, rmark=Mark.ARROW, node2=b)
                            update_flag = True
                            logger.info(f"Orienting Rule 3: {t} *-o {b} to {t} *-> {b}")
                            break
        return pag, update_flag

    def updateList(self, path, new_ts, old_path_list):  # arguments are all lists
        """
        Update the list of paths by adding new paths formed with elements from the given set.
        """
        return old_path_list + [path + [t] for t in new_ts]

    def minDiscrPath(self, a: Node, b: Node, r: Node, pag: MixGraph) -> Optional[List[Node]]:
        """
        Find the minimal discriminating path between two nodes given a third node.
        We had a path a <-* b o-* r and a -> r, then we need to find the minDiscrPath for Rule4 in Zhang 2008.
        Parameters:
        - a: The first node.
        - b: The second node.
        - r: The third node.


        Returns:
        - A list of nodes representing the minimal discriminating path, or None if no such path exists.
        """
        Cand_ts = pag.get_into_nodes(a)  # Get all nodes that point to a 
        visited = {a, b, r}  # Nodes already visited
        Cand_ts = Cand_ts - visited  # Remove visited nodes
        if len(Cand_ts) == 0:
            return None
        
        list_paths = self.updateList([a], Cand_ts, [])  # Initialize paths with a and candidates

        while list_paths:
            path = list_paths.pop(0)
            cand_t = path[-1]  # Last node in the current path

            if self.sepsets.has_sepset(cand_t, r):  # not adjacent
                return path[:: -1] + [b, r]  # t *-> a <-* b o-* r

            pred_t = path[-2]
            visited.add(cand_t)  # Mark the current node as visited

            if pag.has_directed_Edge(cand_t, r) and pag.has_into_Edge(pred_t, cand_t): #  t <-> a <-* b o-* r and t -> r
                Cand_ts = pag.get_into_nodes(cand_t) - visited  # Get unvisited nodes that point to t
                if len(Cand_ts) > 0:
                    list_paths = self.updateList(path, Cand_ts, list_paths)

        return None

    def Rule_4(self, pag: MixGraph, update_flag: bool) -> MixGraph:
        """
        If u = <t, ..., a, b, r> is a discriminating path between t and r for b, and b o-* r;
        then if b ∈ SepSet(t, r), orient b o-* r as b -> r; otherwise orient the triple <a, b, r> as a <-> b <-> r.
        """
        for b, r in pag.get_circ_star_Edge():  # Get edges of the form b o-* r
            # a -> r and b *-> a
            Cand_as = pag.get_parents(r)
            Cand_as = {a for a in Cand_as if pag.has_into_Edge(b, a)}
            while len(Cand_as) > 0:
                a = Cand_as.pop()  # Take one candidate a
                md_path = self.minDiscrPath(a, b, r, pag)
                if md_path is not None:
                    t = md_path[0]
                    if self.sepsets.is_in_sepset(target=b, node1=t, node2=r):
                        pag.update_Edge(node1=b, lmark=Mark.TAIL, rmark=Mark.ARROW, node2=r)
                        
                        logger.info(f"Orienting Rule 4: {b} o-* {r} to {b} -> {r}")
                    else:
                        pag.update_Edge(node1=a, lmark=Mark.ARROW, rmark=Mark.ARROW, node2=b)
                        pag.update_Edge(node1=b, lmark=Mark.ARROW, rmark=Mark.ARROW, node2=r)
                        logger.info(f"Orienting Rule 4: {a} <-> {b} <-> {r}")
                    update_flag = True
                    break  # No need to check other As for this b, r pair
        return pag, update_flag

    def minUncovCircPath(self, path, pag: MixGraph) -> Optional[List[Node]]:
        """
        Find a minimal uncovered circle path starting from the given path([a, r, ..., t, b]).
        Parameters:
            path: [a, r, t, b] under interest, such that r o-o a o-o b o-o t and a, t are not adjacent, b, r are not adjacent.
        """
        
        a = path[0]
        r = path[1]
        t = path[2]
        b = path[3]
        Cand_xs = pag.get_nondirect_adj_nodes(r) # Get all nodes that are adjacent to r by r o-o x.
        visited = {r, a, b, t}
        Cand_xs = Cand_xs - visited  # Remove visited nodes
        if len(Cand_xs) == 0:
            return None

        list_paths = self.updateList([r], Cand_xs, [])  # Initialize paths with r and candidates

        while list_paths:
            path = list_paths.pop(0)
            cand_x = path[-1]
            visited.add(cand_x)
            if pag.has_circ_circ_Edge(cand_x, t): # circle path found
                mpath = [a] + path + [t, b]  # <a, r, ..., x, t, b>
                if self.is_uncovered_path(mpath):
                    return mpath
            else:
                Cand_xis = pag.get_nondirect_adj_nodes(cand_x) # x o-o x'
                Cand_xis = Cand_xis - visited  # Remove visited nodes
                if len(Cand_xis) > 0:
                    list_paths = self.updateList(path, Cand_xis, list_paths)
        return None


    def Rule_5(self, pag: MixGraph, update_flag: bool) -> MixGraph:
        """
        For every (remaining) a o-o b, if there is an uncovered circle path p = ⟨a, r, ..., t, b⟩ between a and b such that 
        a, t are not adjacent and b, r are not adjacent, 
        then orient a o-o b and every edge on p as undirected edges (-).
        """
        for a, b in pag.get_circ_circ_Edge():  # Get edges of the form a o-o b
            Cand_ts = pag.get_nondirect_adj_nodes(b) - {a}  # Get all nodes that are adjacent to b by b o-o t, excluding a
            Cand_ts = {t for t in Cand_ts if (self.sepsets.has_sepset(a, t) and self.sepsets.is_in_sepset(b, a, t))}  # Remove candidates that are adjacent to a
            Cand_rs = pag.get_nondirect_adj_nodes(a) - {b}  # Get all nodes that are adjacent to a by a o-o r, excluding b
            Cand_rs = {r for r in Cand_rs if (self.sepsets.has_sepset(r, b) and self.sepsets.is_in_sepset(a, r, b))}  # Remove candidates that are adjacent to b
            if len(Cand_ts) > 0 and len(Cand_rs) > 0:
                while len(Cand_rs) > 0 and pag.has_circ_circ_Edge(a, b):  # Find a candidate r
                    r = Cand_rs.pop()
                    while len(Cand_ts) > 0 and pag.has_circ_circ_Edge(a, b):  # Find a candidate t
                        t = Cand_ts.pop()
                        if pag.has_circ_circ_Edge(r, t) and self.is_uncovered_path([a, r, t, b]): # the easiest one 
                            pag.update_Edge(node1=a, lmark=Mark.TAIL, rmark=Mark.TAIL, node2=b)  
                            pag.update_Edge(node1=a, lmark=Mark.TAIL, rmark=Mark.TAIL, node2=r)
                            pag.update_Edge(node1=r, lmark=Mark.TAIL, rmark=Mark.TAIL, node2=t)
                            pag.update_Edge(node1=t, lmark=Mark.TAIL, rmark=Mark.TAIL, node2=b)
                            update_flag = True
                            logger.info(f"Orienting Rule 5: There exists an uncovered circle path between {a} o-o {b}, orient {a} - {r} - {t} - {b} and {a} - {b}")
                            
                        else:
                            # Find a minimal uncovered circle path for these r, a, b, and t.
                            ucp_path = self.minUncovCircPath([a, r, t, b], pag) 
                            if ucp_path is not None:
                                # Orient the edges in the uncovered circle path as undirected edges
                                for i in range(len(ucp_path) - 1):
                                    pag.update_Edge(node1=ucp_path[i], lmark=Mark.TAIL, rmark=Mark.TAIL, node2=ucp_path[i + 1])
                                pag.update_Edge(node1=a, lmark=Mark.TAIL, rmark=Mark.TAIL, node2=b)  # Close the circle
                                update_flag = True
                                logger.info(f"Orienting Rule 5: There exists an uncovered circle path between {a} o-o {b}, orient {ucp_path} and {a} - {b}")



        return pag, update_flag

    def Rule_6(self, pag: MixGraph, update_flag: bool) -> MixGraph:
        """
        If a --- b o-* r (a and r may or may not be adjacent), then orient b o-* r as b --* r.
        
        """
        for b, r in pag.get_circ_star_Edge():  # Get edges of the form b o-* r
            Cand_as = pag.get_adj_nodes(b) - {r}
            for a in Cand_as:
                if pag.has_tail_tail_Edge(a, b):
                    pag.update_Edge(node1=b, lmark=Mark.TAIL, rmark=None, node2=r)
                    update_flag = True
                    logger.info(f"Orienting Rule 6: {b} o-* {r} to {b} --* {r}")
                    break  # No need to check other As for this b, r pair
        
        return pag, update_flag

    def Rule_7(self, pag: MixGraph, update_flag: bool) -> MixGraph:
        """
        If a --o b o-* r, and a, r are not adjacent, then orient b o-* r as b --* r.
        """

        for b, r in pag.get_circ_star_Edge():  # Get edges of the form b o-* r
            Cand_as = pag.get_adj_nodes(b) - {r}  # Get all nodes that are adjacent to b by b --o a, excluding r
            for a in Cand_as:
                if pag.has_tail_circ_Edge(a, b) and \
                    (self.sepsets.has_sepset(a, r) and self.sepsets.is_in_sepset(target=b, node1=a, node2=r)):
                    pag.update_Edge(node1=b, lmark=Mark.TAIL, rmark=None, node2=r)
                    update_flag = True
                    logger.info(f"Orienting Rule 7: {b} o-* {r} to {b} --* {r}")
                    break


        return pag, update_flag
        


    def Rule_8(self, pag: MixGraph, update_flag: bool) -> MixGraph:
        """
        If a -> b -> r or a -o b -> r, and a o-> r, orient a o-> r as a -> r.
        """
        for a, r in pag.get_circ_arrow_Edge():
            for b in pag.get_parents(r):
                if pag.has_directed_Edge(a, b) or pag.has_tail_circ_Edge(a, b):
                    pag.update_Edge(node1=a, lmark=Mark.TAIL, rmark=Mark.ARROW, node2=r)
                    update_flag = True
                    logger.info(f"Orienting Rule 8: {a} o-> {r} to {a} -> {r}")
                    break
        return pag, update_flag
    
    def is_uncovered_path(self, path: List[Node]) -> bool:
        """
        Check if the given path is uncovered by sepsets for 'minUncovPdPath' and 'minUncovCircPath' in orient rules.
        In local learning, direct adjacency between nodes may not be fully determined. Therefore, we use self.is_uncovered_path (which relies on separation sets) to check for uncovered paths in orientation rules, rather than MixGraph.is_uncovered_path, to ensure correctness in these cases.
        """
        for i in range(1, len(path) - 1):
            if not (self.sepsets.has_sepset(path[i - 1], path[i + 1]) and self.sepsets.is_in_sepset(target=path[i], node1=path[i - 1], node2=path[i + 1])):
                return False
        return True

    def minUncovPdPath(self, a: Node, b: Node, r: Node, pag: MixGraph) -> Optional[List[Node]]:
        """
            Find a minimal uncovered pd path from initial (a,b,r)
        """
        Cand_ts = pag.get_pd_path_nodes(b) 
        visited = {a, b, r}  # Nodes already visited
        Cand_ts = Cand_ts - visited  # Remove visited nodes
        if len(Cand_ts) == 0:
            return None

        list_paths = self.updateList([b], Cand_ts, [])  # Initialize paths with b and candidates

        while list_paths:
            path = list_paths.pop(0)
            cand_t = path[-1]  # Last node in the current path
            visited.add(cand_t)  # Mark the current node as visited
            tr_edge = pag.get_Edge(cand_t, r)
            if tr_edge is not None and \
                tr_edge.lmark != Mark.ARROW and tr_edge.rmark != Mark.TAIL:
                    mpath = [a] + path + [r] # <a, b, ..., t ,r>
                    if self.is_uncovered_path(mpath):
                        return mpath

            else:
                Cand_tis = pag.get_pd_path_nodes(cand_t)
                Cand_tis = Cand_tis - visited  # Remove visited nodes
                if len(Cand_tis) > 0:
                    list_paths = self.updateList(path, Cand_tis, list_paths)

        return None

    def Rule_9(self, pag: MixGraph, update_flag: bool) -> MixGraph:
        """
        If a o-> r, and p = <a, b, ..., r> is an uncovered potentially directed path from a to r such that r and b are not adjacent, then orient a o-> r as a -> r.
        """
        for a, r in pag.get_circ_arrow_Edge():
            ## find all b s.t. a (o-)--(o>) b and b and r are not connected
            Cand_bs = pag.get_pd_path_nodes(a) # a (o-)--(o>) b
            Cand_bs = {b for b in Cand_bs if not pag.has_edge(b, r)}  # Remove candidates that are adjacent to r
            Cand_bs.remove(r)  # Remove r from candidates
            while len(Cand_bs) > 0:
                b = Cand_bs.pop()
                if not (self.sepsets.has_sepset(b, r) and \
                    self.sepsets.is_in_sepset(target=a, node1=b, node2=r)): 
                    # Verify whether the separation set of b and r has been identified, and check if a is included in the separation set of b and r. Note that a is defined not to be a collider in the triple <b, a, r>.
                    continue
                upd_path = self.minUncovPdPath(a, b, r, pag)
                if upd_path is not None:
                    pag.update_Edge(node1=a, lmark=Mark.TAIL, rmark=Mark.ARROW, node2=r)
                    update_flag = True
                    logger.info(f"Orienting Rule 9: {a} o-> {r} to {a} -> {r} via {b}")
                    break

        return pag, update_flag

    def Rule_10(self, pag: MixGraph, update_flag: bool) -> MixGraph:

        """
        R10: Suppose a o-> r, b -> r <- t, p1 is an uncovered p.d. path from a to b, and p2 is an uncovered p.d. path from a to t.
        Let u be the vertex adjacent to a on p1 (u could be b), and w be the vertex adjacent to a on p2 (w could be t).
        If u and w are distinct, and are not adjacent, then orient a o-> r as a -> r.
        """
        for a, r in pag.get_circ_arrow_Edge():
            pa_r = pag.get_parents(r)  # Get all parents of r
            if len(pa_r) < 2:
                continue
            for b, t in combinations(pa_r, 2):
                ## this is the easiest one
                if pag.has_pd_Edge(a, b) and pag.has_pd_Edge(a, t) and \
                    self.sepsets.has_sepset(b, t) and \
                    self.sepsets.is_in_sepset(target=a, node1=b, node2=t):
                    pag.update_Edge(node1=a, lmark=Mark.TAIL, rmark=Mark.ARROW, node2=r)
                    update_flag = True
                    logger.info(f"Orienting Rule 10: {a} o-> {r} to {a} -> {r} via {b} and {t}")
                    break  # No need to check other pairs of b and t for this a, r pair
                else:
                    Cand_uw = pag.get_pd_path_nodes(a)  # Get all nodes that a (o-)--(o>) can reach
                    Cand_uw.remove(r) # Remove r from candidates
                    if len(Cand_uw) < 2:
                        continue
                    for u, w in combinations(Cand_uw, 2):
                        # for speed
                        if pag.has_edge(u, w):
                            continue  # u and w are adjacent, skip this pair

                        if u == b:
                            p1 = [a, b]
                        else:
                            p1 = self.minUncovPdPath(a, u, b, pag)
                        if p1 is not None:
                            if w == t:
                                p2 = [a, t]
                            else:
                                p2 = self.minUncovPdPath(a, w, t, pag)

                            if p2 is not None and u!=w and \
                                self.sepsets.has_sepset(u, w) and \
                                self.sepsets.is_in_sepset(target=a, node1=u, node2=w):
                                    pag.update_Edge(node1=a, lmark=Mark.TAIL, rmark=Mark.ARROW, node2=r)
                                    update_flag = True
                                    logger.info(f"Orienting Rule 10: {a} o-> {r} to {a} -> {r} via {b} and {t}")
                                    break
                    if pag.has_directed_Edge(a, r):
                        break  # No need to check other pairs of b and t for this a, r pair

        return pag, update_flag


