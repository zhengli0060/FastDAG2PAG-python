from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Union
from fastdag2pag_V2.Utils.Graph_utils import Node
import warnings


@dataclass
class SeparationRecord:
    """
    A single separation-set record.
    """
    sepset: set[str]
    power: Optional[float] = None


class SeparationSet:
    def __init__(self, node_set: set[str]):
        """
        Initialize the SeparationSet with a set of node names (str).
        """
        if not node_set:
            warnings.warn("node_set should be a non-empty set of strings.", UserWarning)

        self.node_set = set(node_set)

        # key: (node1, node2)
        # value: list[SeparationRecord]
        self.sepsets: dict[tuple[str, str], list[SeparationRecord]] = defaultdict(list)

    def _validate_node(self, node: Union[str, Node]) -> str:
        if not isinstance(node, str) and not isinstance(node, Node):
            raise TypeError(f"Node must be a string or Node instance, got {type(node).__name__}.")
        if isinstance(node, Node):
            node = node.name
        if self.node_set and node not in self.node_set:
            raise ValueError(f"Node '{node}' is not in node_set.")
        return node

    def _validate_sepset(self, sepset: set[str]) -> set[str]:
        if not isinstance(sepset, set):
            raise TypeError("sepset must be a set of strings.")
        sepset = set(self._validate_node(node) for node in sepset)
        return sepset


    def _ensure_order(self, node1: Union[str, Node], node2: Union[str, Node]) -> tuple[str, str]:
        """
        Ensure consistent ordering of node pairs.
        """
        node1 = self._validate_node(node1)
        node2 = self._validate_node(node2)

        if node1 == node2:
            raise ValueError("node1 and node2 must be different.")

        return (node1, node2) if node1 < node2 else (node2, node1)

    def add_sepset(
        self,
        node1: Union[str, Node],
        node2: Union[str, Node],
        sepset: set[Union[str, Node]],
        power: Optional[float] = None,
    ) -> None:
        """
        Add a separation set for the given pair of nodes.

        If the same separation set already exists for this node pair,
        do not add a duplicate record. Instead:
        - update its power
        - move it to the end so that it becomes the latest record
        """
        node1, node2 = self._ensure_order(node1, node2)
        sepset = self._validate_sepset(sepset)

        records = self.sepsets[(node1, node2)]

        for i, record in enumerate(records):
            if record.sepset == sepset:
                # Deduplicate: update power and move to the end
                updated_record = SeparationRecord(set(sepset), power)
                records.pop(i)
                records.append(updated_record)
                return

        # Otherwise, append as a new record
        records.append(SeparationRecord(set(sepset), power))


    def has_sepset(self, node1: Union[str, Node], node2: Union[str, Node]) -> bool:
        """
        Check if at least one separation set exists for the given pair of nodes.
        """
        if node1 == node2:
            return False
        node1, node2 = self._ensure_order(node1, node2)
        return (node1, node2) in self.sepsets and len(self.sepsets[(node1, node2)]) > 0

    def get_sepset(self, node1: Union[str, Node], node2: Union[str, Node]) -> Optional[set[str]]:
        """
        Return the latest separation set for the given pair of nodes.
        """
        node1, node2 = self._ensure_order(node1, node2)
        if self.has_sepset(node1, node2):
            return set(self.sepsets[(node1, node2)][-1].sepset)
        return None

    def get_sepset_with_power(
        self,
        node1: Union[str, Node],
        node2: Union[str, Node],
    ) -> Optional[SeparationRecord]:
        """
        Return the latest separation-set record for the given pair of nodes.
        """
        node1, node2 = self._ensure_order(node1, node2)
        if self.has_sepset(node1, node2):
            record = self.sepsets[(node1, node2)][-1]
            return record
        return None

    def get_all_sepset(self, node1: Union[str, Node], node2: Union[str, Node]) -> Optional[list[set[str]]]:
        """
        Return all separation sets for the given pair of nodes.
        """
        node1, node2 = self._ensure_order(node1, node2)
        if self.has_sepset(node1, node2):
            return [set(record.sepset) for record in self.sepsets[(node1, node2)]]
        return None


    def get_all_sepset_with_power(
        self,
        node1: Union[str, Node],
        node2: Union[str, Node],
    ) -> Optional[list[SeparationRecord]]:
        """
        Return all separation-set records for the given pair of nodes.
        """
        node1, node2 = self._ensure_order(node1, node2)
        if self.has_sepset(node1, node2):
            return [
                record
                for record in self.sepsets[(node1, node2)]
            ]
        return None

    def is_in_sepset(self, target: Union[str, Node], node1: Union[str, Node], node2: Union[str, Node]) -> bool:
        """
        Check if target is in the latest separation set of the given pair.

        Returns False if no separation set exists.
        """
        target = self._validate_node(target)
        latest_sepset = self.get_sepset(node1, node2)
        if latest_sepset is None:
            return False
        return target in latest_sepset

    def is_in_sepset_stable(
        self,
        target: Union[str, Node],
        node1: Union[str, Node],
        node2: Union[str, Node],
        mode: str = "all",
    ) -> Optional[bool]:
        """
        Stable version based on all stored separation sets.

        mode:
        - "all": target must be in all separation sets
        - "any": target must be in at least one separation set
        """
        target = self._validate_node(target)
        all_sepsets = self.get_all_sepset(node1, node2)
        if all_sepsets is None:
            return None

        if mode == "all":
            return all(target in s for s in all_sepsets)
        elif mode == "any":
            return any(target in s for s in all_sepsets)
        else:
            raise ValueError("mode must be either 'all' or 'any'.")

    def __repr__(self) -> str:
        result = []
        for (node1, node2), records in self.sepsets.items():
            formatted = [
                {"sepset": record.sepset, "power": record.power}
                for record in records
            ]
            result.append(f"{node1} - {node2}: {formatted}")
        return "\n".join(result)


if __name__ == "__main__":
    node_set = {"A", "B", "C", "D", "E"}
    sepset_store = SeparationSet(node_set)

    sepset_store.add_sepset("A", "B", {"C"}, power=0.81)
    sepset_store.add_sepset("A", "B", {"C", "D"}, power=0.92)
    sepset_store.add_sepset("A", "B", {"C"}, power=0.95)   # duplicate sepset, update power and move to latest
    sepset_store.add_sepset("A", "C", {"D"}, power=0.77)

    print("All stored records:")
    print(sepset_store)
    print()

    print("Latest sepset(A, B):", sepset_store.get_sepset("A", "B"))
    print("Latest record(A, B):", sepset_store.get_sepset_with_power("A", "B"))
    print()

    print("All sepsets(A, B):", sepset_store.get_all_sepset("A", "B"))
    print("All records(A, B):", sepset_store.get_all_sepset_with_power("A", "B"))
    print("All records(A, B):", sepset_store.get_all_sepset_with_power("A", "B"))
    print()

    print("Is C in latest sepset(A, B)?", sepset_store.is_in_sepset("C", "A", "B"))
    print("Is D in any sepset(A, B)?", sepset_store.is_in_sepset_stable("D", "A", "B", mode="any"))
    print("Is C in all sepsets(A, B)?", sepset_store.is_in_sepset_stable("C", "A", "B", mode="all"))