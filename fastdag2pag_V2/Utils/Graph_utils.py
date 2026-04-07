from enum import Enum
from typing import Optional, Tuple
from dataclasses import dataclass
from collections import namedtuple

class Mark(Enum):
    """pcalg package mark definition"""
    TAIL = 3   # "-"
    ARROW = 2  # ">"
    CIRCLE = 1 # "o"
    NULL = 0   # " "

    def __eq__(self, other):
        if isinstance(other, Mark):
            return self.value == other.value
        return False

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return {
            Mark.TAIL: "-",
            Mark.ARROW: ">",
            Mark.CIRCLE: "o",
            Mark.NULL: " ",
        }[self]

    def __repr__(self):
        return f"Mark.{self.name}"

Pattern = Tuple[Mark, Mark]

@dataclass(frozen=True, eq=True)
class Node:
    name: Optional[str] = None
    index: Optional[int] = None

    def __repr__(self):
        return str(self.name)

    def __str__(self):
        return str(self.name)


@dataclass(frozen=True, eq=True)
class Edge:
    start: Node
    lmark: Mark
    rmark: Mark
    end: Node

    def __repr__(self):
        rmark_symbol = {Mark.TAIL: "-", Mark.ARROW: ">", Mark.CIRCLE: "o"}
        lmark_symbol = {Mark.TAIL: "-", Mark.ARROW: "<", Mark.CIRCLE: "o"}
        return f"{self.start}{lmark_symbol[self.lmark]}-{rmark_symbol[self.rmark]} {self.end}" 
    
    def __str__(self):
        return self.__repr__()
