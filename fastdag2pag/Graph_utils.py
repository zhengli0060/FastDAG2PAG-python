from enum import Enum
from typing import Optional
class Mark(Enum):
    "causal-learn package mark definition"
    # TAIL = -1
    # ARROW = 1
    # CIRCLE = 2
    # NULL = 0


    "pcalg package mark definition"
    TAIL = 3   # "-"
    ARROW = 2  # ">"
    CIRCLE = 1 # "o"
    NULL = 0   # " "

    def __eq__(self, other):
        if isinstance(other, Mark):
            return self.value == other.value
        return NotImplemented

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return {Mark.TAIL: "-", Mark.ARROW: ">", Mark.CIRCLE: "o", Mark.NULL: " "}[self]

    def __repr__(self):
        return f"Mark.{self.name}"


from dataclasses import dataclass

@dataclass(frozen=True)
class Node:
    name: Optional[str] = None
    index: Optional[int] = None

    def __repr__(self):
        return f"{self.name}"
    def __str__(self):
        return f"{self.name}"
    # def __eq__(self, value):
    #     if not isinstance(value, Node):
    #         return NotImplemented
    #     return self.name == value.name 

    
class Edge:
    def __init__(self, start: Node, lmark: Mark, rmark: Mark, end: Node):  # start lmark-rmark end
        self.start = start
        self.end = end
        self.lmark = lmark  # The mark here can be Mark.TAIL, Mark.ARROW, or Mark.CIRCLE
        self.rmark = rmark
    def __repr__(self):
        rmark_symbol = {Mark.TAIL: "-", Mark.ARROW: ">", Mark.CIRCLE: "o"}
        lmark_symbol = {Mark.TAIL: "-", Mark.ARROW: "<", Mark.CIRCLE: "o"}
        return f"{self.start}{lmark_symbol[self.lmark]}-{rmark_symbol[self.rmark]} {self.end}"
    def _invert(self): # 反转边
        # 反转边的方向
        self.start, self.end = self.end, self.start
        # 交换标记
        self.lmark, self.rmark = self.rmark, self.lmark

    def copy(self):
        return Edge(self.start, self.lmark, self.rmark, self.end)

        

if __name__ == "__main__":

    node1 = Node("A", 0)
    node2 = Node("B", 1)
    node3 = Node("C", 2)


    edge1 = Edge(node1, Mark.ARROW, Mark.TAIL, node2)
    edge2 = Edge(node2, Mark.CIRCLE, Mark.CIRCLE, node3)

    print(edge1) 
    print(edge2)  



