from collections import defaultdict, deque
from itertools import combinations
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union, Any
import numpy as np
import pandas as pd
from fastdag2pag_V2.Utils.Graph_utils import Mark, Node, Pattern, Edge


class PMG:
    """
    A Partial Mixed Graph implementation optimized for edge-type-aware queries.
    - Nodes are stored in a list for ordered access and a dict for quick lookup.
    - Edges are stored in an adjacency dict with patterns, and a pattern index for O(1) retrieval of neighbors by edge type.
    """

    def __init__(self, incoming_graph_data: Optional[Union[pd.DataFrame, np.ndarray, list]] = None):


        self.Node_list: List[Node] = []
        self._str_Node: Dict[str, Node] = {}

        # _adj[u][v] = (mark_at_u, mark_at_v)
        self._adj: Dict[Node, Dict[Node, Pattern]] = defaultdict(dict)
        
        # _pattern_index[u][(mark_at_u, mark_at_v)] = {neighbors}
        self._pattern_index: Dict[Node, Dict[Pattern, Set[Node]]] = defaultdict(lambda: defaultdict(set))
        
        if incoming_graph_data is not None:
            if isinstance(incoming_graph_data, pd.DataFrame):
                self.from_pandas_adjacency(incoming_graph_data)
            elif isinstance(incoming_graph_data, np.ndarray):
                self.from_numpy_array(incoming_graph_data)
            elif isinstance(incoming_graph_data, list):
                self.from_node_list(incoming_graph_data)
            else:
                raise TypeError("Invalid graph data type.")

    # ------------------------------------------------------------------
    # Base helper methods
    # ------------------------------------------------------------------
    def _init_Node_edges(self, str_list: List[str]):

        self.Node_list = [Node(name=s, index=i) for i, s in enumerate(str_list)]
        self._str_Node = {node.name: node for node in self.Node_list}
        for node in self.Node_list:
            _ = self._adj[node]
            _ = self._pattern_index[node]


    def __contains__(self, node: Node) -> bool:
        return self.has_Node(node)

    def __iter__(self) -> Iterator[Node]:
        return iter(self.Node_list)

    def __len__(self) -> int:
        return len(self.Node_list)

    def __eq__(self, other: "PMG") -> bool:
        if not isinstance(other, PMG): return False
        if len(self.Node_list) != len(other.Node_list): return False
        
        for u, nbrs in self._adj.items():
            if u not in other._adj: return False
            for v, marks in nbrs.items():
                if other._adj[u].get(v) != marks:
                    return False
        return True


    def _node_order_key(self, node: Node) -> Tuple[int, str]:
        assert isinstance(node, Node), "Expected node to be of type Node."
        return node.index
    
    def clear(self) -> None:
        self.Node_list.clear()
        self._adj.clear()
        self._pattern_index.clear()

    def _is_canonical_order(self, node1: Node, node2: Node) -> bool:
        return self._node_order_key(node1) < self._node_order_key(node2)

    # ------------------------------------------------------------------
    # Node add/remove/update
    # ------------------------------------------------------------------
    def add_Node(self, node: Node) -> None:
        if node not in self.Node_list:
            self.Node_list.append(node)
            self.Node_list.sort(key=self._node_order_key)
            _ = self._adj[node]
            _ = self._pattern_index[node]

    def add_Nodes_from(self, nodes: Iterable[Node]) -> None:
        changed = False
        for node in nodes:
            if not isinstance(node, Node):
                raise TypeError("All nodes must be of type Node.")
            if node not in self.Node_list:
                self.Node_list.append(node)
                changed = True
        if changed:
            self.Node_list.sort(key=self._node_order_key)

    def remove_Node(self, node: Node) -> None:
        if not self.has_Node(node):
            raise ValueError(f"Node {node} does not exist in the graph.")
        neighbors = list(self._adj[node].keys())
        for nbr in neighbors:
            self._remove_edge_internal(node, nbr)
        self._adj.pop(node, None)
        self._pattern_index.pop(node, None)
        self.Node_list = [n for n in self.Node_list if n != node]

    def has_Node(self, node: Node) -> bool:
        return node in self.Node_list

    def nodes(self) -> List[Node]:
        return self.Node_list.copy()


    # ------------------------------------------------------------------
    # Edge add/remove/update (core optimizations)
    # ------------------------------------------------------------------

    def _insert_edge(self, u: Node, v: Node, mark_u: Mark, mark_v: Mark) -> None:

        assert u != v, "Self-loops are not allowed in PMG."
        assert self.has_Node(u) and self.has_Node(v), f"Both nodes must exist in the graph to add an edge. Missing: {u if not self.has_Node(u) else ''} {v if not self.has_Node(v) else ''}"
        assert not self.has_edge(u, v), f"Edge between {u} and {v} already exists. Use update_edge to modify existing edges."

        self._adj[u][v] = (mark_u, mark_v)
        self._adj[v][u] = (mark_v, mark_u)
        self._pattern_index[u][(mark_u, mark_v)].add(v)
        self._pattern_index[v][(mark_v, mark_u)].add(u)

    def add_circ_edge(self, u: Node, v: Node): self._insert_edge(u, v, Mark.CIRCLE, Mark.CIRCLE)
    def add_directed_edge(self, u: Node, v: Node): self._insert_edge(u, v, Mark.TAIL, Mark.ARROW)
    def add_bidirected_edge(self, u: Node, v: Node): self._insert_edge(u, v, Mark.ARROW, Mark.ARROW)
    def add_circ_arrow_edge(self, u: Node, v: Node): self._insert_edge(u, v, Mark.CIRCLE, Mark.ARROW)
    def add_circ_tail_edge(self, u: Node, v: Node): self._insert_edge(u, v, Mark.CIRCLE, Mark.TAIL)
    def add_tail_edge(self, u: Node, v: Node): self._insert_edge(u, v, Mark.TAIL, Mark.TAIL)

    def add_edges_from(self, ebunch: Iterable[Tuple]):
        for item in ebunch:
            if len(item) == 3 and isinstance(item[2], dict) and "edge" in item[2]:
                self._insert_edge(item[0], item[1], item[2]["edge"].lmark, item[2]["edge"].rmark)
            elif len(item) == 3 and hasattr(item[2], 'lmark'):
                self._insert_edge(item[0], item[1], item[2].lmark, item[2].rmark)
            else:
                raise ValueError("Unsupported edge tuple format in add_edges_from.")    
            

    def _remove_edge_internal(self, u: Node, v: Node) -> None:
        if v not in self._adj[u]: return
        mark_u, mark_v = self._adj[u][v]
        
        self._pattern_index[u][(mark_u, mark_v)].discard(v)
        self._pattern_index[v][(mark_v, mark_u)].discard(u)
        
        del self._adj[u][v]
        del self._adj[v][u]

    def remove_edge(self, node1: Node, node2: Node) -> None:
        if not self.has_edge(node1, node2):
            raise ValueError(f"Edge between {node1} and {node2} does not exist.")
        self._remove_edge_internal(node1, node2)        

    def remove_edges_from(self, ebunch: Iterable[Tuple[Node, Node]]) -> None:
        for u, v in list(ebunch):
            if self.has_edge(u, v):
                self._remove_edge_internal(u, v)

    def clear_all_edges(self) -> None:
        self._adj.clear()
        self._pattern_index.clear()
        for node in self.Node_list:
            self._adj[node] = {}
            self._pattern_index[node] = defaultdict(set)    


    def update_edge(self, node1: Node, lmark: Mark, rmark: Mark, node2: Node):
        if not self.has_edge(node1, node2):
            raise ValueError(f"Edge does not exist.")
        
        old_marks = self._adj[node1][node2]
        new_lmark = old_marks[0] if lmark is None else lmark
        new_rmark = old_marks[1] if rmark is None else rmark
        
        self._remove_edge_internal(node1, node2)
        self._insert_edge(node1, node2, new_lmark, new_rmark)


    def clear_all_orientations(self):
        for u, nbrs in list(self._adj.items()):
            for v in list(nbrs.keys()):
                if self._is_canonical_order(u, v):
                    self._remove_edge_internal(u, v)
                    self._insert_edge(u, v, Mark.CIRCLE, Mark.CIRCLE)


    def has_edge(self, node1: Node, node2: Node) -> bool:
        return self.has_Node(node1) and (node2 in self._adj[node1])

    def edges(self, nbunch: Optional[Node] = None, data: bool = False):
        result = []
        if nbunch is None:
            for u, nbrs in self._adj.items():
                for v, marks in nbrs.items():
                    if self._is_canonical_order(u, v):
                        if data:
                            result.append((u, v, {"edge": Edge(u, marks[0], marks[1], v)}))
                        else:
                            result.append((u, v))
        else:
            if not self.has_Node(nbunch):
                raise ValueError(f"Node {nbunch} does not exist.")
            for nbr, marks in self._adj.get(nbunch, {}).items():
                if data:
                    result.append((nbunch, nbr, {"edge": Edge(nbunch, marks[0], marks[1], nbr)}))
                else:
                    result.append((nbunch, nbr))
        return result

    def degree(self, nbunch: Optional[Node] = None):
        if nbunch is None:
            return [(node, len(self._adj[node])) for node in self.Node_list]
        if not self.has_Node(nbunch):
            raise ValueError(f"Node {nbunch} does not exist in the graph.")
        return len(self._adj[nbunch])

    def avg_degree(self) -> float:
        total_degree = sum(len(self._adj[node]) for node in self.Node_list)
        return total_degree / len(self.Node_list) if self.Node_list else 0.0

    def max_degree(self) -> int:
        return max((len(self._adj[node]) for node in self.Node_list), default=0)




    # ------------------------------------------------------------------
    # Fast edge property queries O(1)
    # ------------------------------------------------------------------
    def get_Edge(self, start: Node, end: Node) -> Optional[Edge]:
        marks = self._adj.get(start, {}).get(end)
        return Edge(start, marks[0], marks[1], end) if marks else None

    def has_directed_edge(self, start: Node, end: Node) -> bool:
        return self._adj.get(start, {}).get(end) == (Mark.TAIL, Mark.ARROW)

    def has_bidirected_edge(self, start: Node, end: Node) -> bool:
        return self._adj.get(start, {}).get(end) == (Mark.ARROW, Mark.ARROW)

    def has_into_edge(self, start: Node, end: Node) -> bool:
        marks = self._adj.get(start, {}).get(end)
        return marks is not None and marks[1] == Mark.ARROW

    def has_pd_edge(self, start: Node, end: Node) -> bool:
        marks = self._adj.get(start, {}).get(end)
        return marks is not None and (marks[0] in {Mark.CIRCLE, Mark.TAIL}) and (marks[1] in {Mark.CIRCLE, Mark.ARROW})

    def has_out_edge(self, start: Node, end: Node) -> bool:
        marks = self._adj.get(start, {}).get(end)
        return marks is not None and marks[0] == Mark.TAIL

    def has_circ_star_edge(self, start: Node, end: Node) -> bool:
        marks = self._adj.get(start, {}).get(end)
        return marks is not None and marks[0] == Mark.CIRCLE

    def has_tail_circ_edge(self, start: Node, end: Node) -> bool:
        return self._adj.get(start, {}).get(end) == (Mark.TAIL, Mark.CIRCLE)

    def has_tail_tail_edge(self, start: Node, end: Node) -> bool:
        return self._adj.get(start, {}).get(end) == (Mark.TAIL, Mark.TAIL)

    def has_circ_circ_edge(self, start: Node, end: Node) -> bool:
        return self._adj.get(start, {}).get(end) == (Mark.CIRCLE, Mark.CIRCLE)

    def has_circ_arrow_edge(self, start: Node, end: Node) -> bool:
        return self._adj.get(start, {}).get(end) == (Mark.CIRCLE, Mark.ARROW)

    # ------------------------------------------------------------------
    # Edge collection queries
    # ------------------------------------------------------------------
    def get_circ_star_edge(self) -> List[tuple[Node, Node]]:
        edges = []
        for u, nbrs in self._adj.items():
            for v, marks in nbrs.items():
                if self._is_canonical_order(u, v):
                    if marks[0] == Mark.CIRCLE: edges.append((u, v))
                    if marks[1] == Mark.CIRCLE: edges.append((v, u))
        return edges

    def get_circ_circ_edge(self) -> List[tuple[Node, Node]]:
        edges = []
        for u, nbrs in self._adj.items():
            for v, marks in nbrs.items():
                if self._is_canonical_order(u, v) and marks == (Mark.CIRCLE, Mark.CIRCLE):
                    edges.append((u, v))
        return edges

    def get_circ_arrow_edge(self) -> List[tuple[Node, Node]]:
        edges = []
        for u, nbrs in self._adj.items():
            for v, marks in nbrs.items():
                if self._is_canonical_order(u, v):
                    if marks == (Mark.CIRCLE, Mark.ARROW): edges.append((u, v))
                    elif marks == (Mark.ARROW, Mark.CIRCLE): edges.append((v, u))
        return edges

    def get_directed_edge(self) -> List[tuple[Node, Node]]:
        edges = []
        for u, nbrs in self._adj.items():
            for v, marks in nbrs.items():
                if self._is_canonical_order(u, v):
                    if marks == (Mark.TAIL, Mark.ARROW): edges.append((u, v))
                    elif marks == (Mark.ARROW, Mark.TAIL): edges.append((v, u))
        return edges

    # ------------------------------------------------------------------
    # Node set queries
    # ------------------------------------------------------------------


    def _neighbors_by_pattern(self, node: Node, pattern: Pattern) -> Set[Node]:
        return self._pattern_index.get(node, {}).get(pattern, set())

    def get_adj_nodes(self, node: Node) -> Set[Node]:
        return set(self._adj.get(node, {}).keys())

    def get_into_nodes(self, node: Node) -> Set[Node]:
        """
        Get the set of nodes that are adjacent into the given node in the graph. 
        node <-*
        """
        idx = self._pattern_index.get(node, {})
        return set(idx.get((Mark.ARROW, Mark.TAIL), set()) |
                   idx.get((Mark.ARROW, Mark.ARROW), set()) |
                   idx.get((Mark.ARROW, Mark.CIRCLE), set()))

    def get_no_into_nodes(self, node: Node) -> Set[Node]:
        """
        Get the set of nodes that are adjacent to the given node but not into it. 
        node -* or node o-*
        """
        return self.get_adj_nodes(node) - self.get_into_nodes(node)

    def get_nondirect_adj_nodes(self, node: Node) -> Set[Node]:
        """
        Get the set of nodes that are adjacent to the given node with non-directed edges.
        node o-o
        """
        return set(self._neighbors_by_pattern(node, (Mark.CIRCLE, Mark.CIRCLE)))

    def get_circ_star_nodes(self, node: Node) -> Set[Node]:
        """
        Get the set of nodes that are adjacent to the given node with circle-star patterns.
        node o-* or node o-o
        """
        idx = self._pattern_index.get(node, {})
        return set(idx.get((Mark.CIRCLE, Mark.CIRCLE), set()) |
                   idx.get((Mark.CIRCLE, Mark.ARROW), set()) |
                   idx.get((Mark.CIRCLE, Mark.TAIL), set()))

    def get_pd_path_nodes(self, node: Node) -> Set[Node]:
        """
        Get the set of nodes that are adjacent to the given node with patterns that allow potential directed paths.
        node o-* or node -* 
        """

        idx = self._pattern_index.get(node, {})
        return set(idx.get((Mark.CIRCLE, Mark.CIRCLE), set()) |
                   idx.get((Mark.CIRCLE, Mark.ARROW), set()) |
                   idx.get((Mark.TAIL, Mark.CIRCLE), set()) |
                   idx.get((Mark.TAIL, Mark.ARROW), set()))

    def get_parents(self, node: Node) -> Set[Node]:
        """
        Get the set of nodes that are parents of the given node in the graph.
        node <- 
        """
        return set(self._neighbors_by_pattern(node, (Mark.ARROW, Mark.TAIL)))

    def get_children(self, node: Node) -> Set[Node]:
        """
        Get the set of nodes that are children of the given node in the graph.
        node -> 
        """
        return set(self._neighbors_by_pattern(node, (Mark.TAIL, Mark.ARROW)))

    def get_spouse(self, node: Node) -> Set[Node]:
        """
        Get the set of nodes that are spouses of the given node in the graph.
        node <->
        """
        return set(self._neighbors_by_pattern(node, (Mark.ARROW, Mark.ARROW)))

    def get_district(self, node: Node) -> Set[Node]:
        """
        Get the district of the given node in the graph.  
        The district includes all nodes reachable from the given node via only bidirected edges (<->).
        """
        district: Set[Node] = set()
        stack = [node]
        visited: Set[Node] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            district.add(current)
            stack.extend(self.get_spouse(current) - visited)
        district.discard(node)
        return district

    def get_PossibleDe(self, node: Union[str, int]) -> Set[Union[str, int]]:
        """
        A possibly directed path or possibly causal path from X to Y is a path from X to Y that 
        does not contain an arrowhead pointing in the direction of X.
        If there is a directed (possibly directed) path from X to Y, then X is a ancestor (possible ancestor) of Y, 
        and Y is a descendant (possible descendant) of X.
        """
        possible_de = set()
        stack = [node]
        visited = set()
        while stack:
            current_node = stack.pop()
            if current_node in visited:
                continue
            visited.add(current_node)
            possible_de.add(current_node)
            stack.extend(self.get_no_into_nodes(current_node) - visited)
        possible_de.discard(node)
        return possible_de

    def max_pds_size(self) -> int:
        return max((len(self.get_possible_d_sep(node)) for node in self.Node_list), default=0)

    def get_possible_d_sep(self, node: Node) -> Set[Node]:
        """
        Compute Possible-D-SEP(node):
        X_k ∈ pds(C, X_i, X_j) iff there exists a path π between X_i and X_k such that
        for every subpath <X_m, X_l, X_h> of π, X_l is a collider on the subpath in C
        or <X_m, X_l, X_h> is a triangle in C (i.e. X_m and X_h are adjacent).
        Here we return the set of nodes X_k reachable from `node` satisfying that condition.
        """
        results: Set[Node] = set()
        for neigh in self.get_adj_nodes(node):
            if neigh == node:
                continue
            stack: List[List[Node]] = [[node, neigh]]
            while stack:
                path = stack.pop()
                last = path[-1]
                if last is not node:
                    results.add(last)
                for nbr in self.get_adj_nodes(last):
                    if nbr in path:
                        continue
                    prev = path[-2]
                    curr = last
                    if self.is_collider(prev, curr, nbr) or self.has_edge(prev, nbr):
                        stack.append(path + [nbr])
        results.discard(node)
        return results

    # ------------------------------------------------------------------
    # Path search and complex validation algorithms (removed unnecessary deepcopy and checks)
    # ------------------------------------------------------------------
    def has_path(self, source: Node, end: Node) -> bool:
        if source == end: return True
        visited = {source}
        q = deque([source])
        while q:
            cur = q.popleft()
            for nbr in self._adj[cur].keys():
                if nbr == end: return True
                if nbr not in visited:
                    visited.add(nbr)
                    q.append(nbr)
        return False

    def has_pd_path(self, source: Node, end: Node) -> bool:
        visited: Set[Node] = set()
        def dfs(current: Node) -> bool:
            if current == end: return True
            visited.add(current)
            for neighbor in self.get_pd_path_nodes(current):
                if neighbor not in visited and dfs(neighbor):
                    return True
            return False
        return dfs(source)

    def has_directed_path(self, source: Node, end: Node) -> bool:
        visited: Set[Node] = set()
        def dfs(current: Node) -> bool:
            if current == end: return True
            visited.add(current)
            for neighbor in self.get_children(current):
                if neighbor not in visited and dfs(neighbor):
                    return True
            return False
        return dfs(source)

    def get_all_paths(self, source: Node, end: Node) -> List[List[Node]]:
        results = []
        def dfs(current, path, visited):
            if current == end:
                results.append(list(path))
                return
            for nbr in self._adj[current]:
                if nbr not in visited:
                    visited.add(nbr)
                    path.append(nbr)
                    dfs(nbr, path, visited)
                    path.pop()
                    visited.remove(nbr)
        dfs(source, [source], {source})
        return results

    def get_all_uncovered_pd_path(self, source: Node, end: Node) -> List[List[Node]]:
        results = []
        def dfs(path, visited):
            current = path[-1]
            if current == end:
                results.append(list(path))
                return
            for neighbor in self.get_pd_path_nodes(current):
                if neighbor in visited: continue
                if len(path) >= 2 and neighbor in self._adj[path[-2]]: continue
                
                path.append(neighbor)
                visited.add(neighbor)
                dfs(path, visited)
                visited.remove(neighbor)
                path.pop()

        dfs([source], {source})
        return results

    def get_all_uncovered_collider_paths_from_target(self, source: Node) -> List[List[Node]]:
        results = []
        def dfs(path: List[Node], visited: Set[Node]):
            current = path[-1]
            prev = path[-2]
            extended = False
            for neighbor in self.get_adj_nodes(current):
                if neighbor in visited: continue
                if not self.has_into_Edge(neighbor, current): continue
                if neighbor in self._adj[prev]: continue

                path.append(neighbor)
                if len(path) >= 3 and list(path) not in results:
                    results.append(list(path))

                if self.has_into_Edge(current, neighbor):
                    visited.add(neighbor)
                    dfs(path, visited)
                    visited.remove(neighbor)
                path.pop()
                extended = True
            return extended

        candidate_paths = [[source, neighbor] for neighbor in self.get_adj_nodes(source) if self.has_into_Edge(source, neighbor)]
        for path in candidate_paths:
            dfs(path, set(path))
        return results

    def is_collider(self, node1: Node, node2: Node, node3: Node) -> bool:
        return self.has_into_Edge(node1, node2) and self.has_into_Edge(node3, node2)

    def is_collider_path(self, path: List[Node]) -> bool:
        if len(path) < 3: return False
        return all(self.is_collider(path[i-1], path[i], path[i+1]) for i in range(1, len(path)-1))

    def is_uncovered_path(self, path: List[Node]) -> bool:
        if len(path) < 3: return True
        return all(path[i+1] not in self._adj.get(path[i-1], {}) for i in range(1, len(path)-1))

    def is_potential_directed_path(self, path: List[Node]) -> bool:
        if len(path) < 2: return False
        for i in range(len(path) - 1):
            marks = self._adj.get(path[i], {}).get(path[i+1])
            if marks is None or marks[0] == Mark.ARROW or marks[1] == Mark.TAIL:
                return False
        return True

    def visible_Edge(self, X: Node, Y: Node) -> bool:
        if not self.has_directed_Edge(X, Y):
            return False

        # 1) V not adjacent to Y with V *-> X
        for V in self.get_into_nodes(X):
            if not self.has_edge(V, Y):
                return True

        # 2) collider path from V to X into X, every non-endpoint node is a parent of Y
        parents_Y = self.get_parents(Y)
        district_X = self.get_district(X)
        discriminator = parents_Y & district_X

        cand_vs: Set[Node] = set()
        for node in discriminator:
            S = self.get_into_nodes(node) - ({X} | discriminator)
            for s in S:
                if not self.has_edge(s, Y):
                    cand_vs.add(s)

        for V in cand_vs:
            for path in self.get_all_paths(source=V, end=X):
                if all(n in discriminator for n in path[1:-1]) and self.is_collider_path(path):
                    return True
        return False

    def find_unique_triplets(self) -> List[Tuple[Node, Node, Node]]:
        """
        Efficiently find unique triplets <z, y, x> in MixGraph, avoiding symmetric duplicates.
        z.index < y.index < x.index, avoid <z, y, x> and <x, y, z>
        Returns:
            List [Tuple[Node, Node, Node]]: List of unique triplets. 
        """
        triplets = []
        for y in self.Node_list:
            neighbors_y = list(self.get_adj_nodes(y))
            if len(neighbors_y) >= 2:
                for z, x in combinations(neighbors_y, 2):
                    triplets.append((z, y, x))
        return triplets

    def _init_complete_graph(self):
        for node1, node2 in combinations(self.Node_list, 2):
            if not self.has_edge(node1, node2):
                self.add_circ_edge(node1, node2)

    # ------------------------------------------------------------------
    # Data import
    # ------------------------------------------------------------------
    def from_numpy_array(self, adj_matrix: np.ndarray, var_list: List[str] = None):
        self.clear()
        n = adj_matrix.shape[0]
        var_list = list(var_list) if var_list else [i for i in range(n)]
        self._init_Node_edges(var_list)
            
        for i in range(n):
            for j in range(i + 1, n):
                mark_j = adj_matrix[i, j]  # i - mark j
                mark_i = adj_matrix[j, i]  # j - mark i
                if mark_j == Mark.NULL.value and mark_i == Mark.NULL.value:
                    continue
                self._insert_edge(self.Node_list[i], self.Node_list[j], Mark(mark_i), Mark(mark_j))

    def DAG_from_numpy_array(self, adj_matrix: np.ndarray, var_list: List[str] = None):
        self.clear()
        n = adj_matrix.shape[0]
        var_list = list(var_list) if var_list else [i for i in range(n)]
        self._init_Node_edges(var_list)
            
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j] == 1:
                    self._insert_edge(self.Node_list[i], self.Node_list[j], Mark.TAIL, Mark.ARROW)
                elif adj_matrix[j, i] == 1:
                    self._insert_edge(self.Node_list[i], self.Node_list[j], Mark.ARROW, Mark.TAIL)

    def from_pandas_adjacency(self, adj_matrix: pd.DataFrame, graph_type: str = 'PMG'):
        var_list = adj_matrix.columns.tolist()
        if graph_type == 'PMG':
            self.from_numpy_array(adj_matrix.to_numpy(), var_list=var_list)
        elif graph_type == 'DAG':
            self.DAG_from_numpy_array(adj_matrix.to_numpy(), var_list=var_list)
        else:
            raise ValueError("graph_type must be 'PMG' or 'DAG'.")

    def from_node_list(self, var_list: List[Union[str, Node]]):
        self.clear()
        if all(isinstance(node, Node) for node in var_list):
            self.Node_list = var_list
            self._str_Node = {node.name: node for node in self.Node_list}
            for node in self.Node_list:
                _ = self._adj[node]
                _ = self._pattern_index[node]
        elif all(isinstance(node, str) for node in var_list):
            self._init_Node_edges(var_list)
        else:
            raise ValueError("All items in var_list must be either Node instances or strings.")

    # ------------------------------------------------------------------
    # Graph comparison and copy
    # ------------------------------------------------------------------
    def _to_numpy_array(self) -> np.ndarray:
        length = len(self.Node_list)
        graph_matrix = np.zeros((length, length), dtype=int)
        for u, nbrs in self._adj.items():
            for v, marks in nbrs.items():
                if self._is_canonical_order(u, v):
                    graph_matrix[u.index, v.index] = marks[1].value
                    graph_matrix[v.index, u.index] = marks[0].value
        return graph_matrix

    def _to_pandas_adjacency(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._to_numpy_array(),
            index=[node.name for node in self.Node_list],
            columns=[node.name for node in self.Node_list],
        )



    def copy(self) -> "PMG":
        new_graph = PMG()
        new_graph.Node_list = self.Node_list.copy()
        new_graph._str_Node = self._str_Node.copy()

        for u, nbrs in self._adj.items():
            new_graph._adj[u] = nbrs.copy()
            for v, marks in nbrs.items():
                new_graph._pattern_index[u][marks].add(v)
                
        return new_graph

    # ------------------------------------------------------------------
    # Visualization utilities
    # ------------------------------------------------------------------

    def to_pydot(self, filename: str = None, view: bool = True, **kwargs):
        import sys
        if not (sys.version_info.major == 3 and sys.version_info.minor == 9 and sys.version_info.micro == 19):
            raise RuntimeError("Python version must be 3.9.19")

        import pydot

        graph_pydot = pydot.Dot(graph_type='digraph', fontsize=18)
        target = kwargs.get('target', None)
        Mb_nodes = kwargs.get('Mb_nodes', [])
        Mb_names = [node.name for node in Mb_nodes] if Mb_nodes is not None else None
        latent_nodes = kwargs.get('latent_nodes', [])
        selection_bias_nodes = kwargs.get('selection_bias_nodes', [])
        for node in self.Node_list:
            node_name = str(node.name)
            if node_name in latent_nodes:
                graph_pydot.add_node(pydot.Node(node_name, shape='circle', style='filled', color='lightgray'))
            elif node_name in selection_bias_nodes:
                graph_pydot.add_node(pydot.Node(node_name, shape='box', style='filled', color='lightgray'))
            else:
                if target is not None and node.name == target.name:
                    graph_pydot.add_node(pydot.Node(node_name, shape='circle', style='filled', fillcolor='#FF6666CC', color='#CC0000'))
                elif node.name in Mb_names:
                    graph_pydot.add_node(pydot.Node(node_name, shape='circle', style='filled', fillcolor='#66B3FFCC', color='#0055AA'))
                else:
                    graph_pydot.add_node(pydot.Node(node_name, shape='circle', style='', color='black'))

        map_mark = {Mark.CIRCLE: "odot", Mark.TAIL: "none", Mark.ARROW: "normal"}
        for u, nbrs in self._adj.items():
            for v, marks in nbrs.items():
                if self._is_canonical_order(u, v):
                    graph_pydot.add_edge(
                        pydot.Edge(
                            str(u.name),
                            str(v.name),
                            arrowtail=map_mark[marks[0]],
                            arrowhead=map_mark[marks[1]],
                            dir='both',
                        )
                    )

        if filename is not None:
            pdf_path = filename + '.pdf'
            graph_pydot.write_pdf(pdf_path)
            if view:
                import os
                os.startfile(pdf_path)
        return graph_pydot