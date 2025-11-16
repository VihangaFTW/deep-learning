class Node:
    def __init__(self, data: float, _children: tuple[Node, ...]= (), _op:str = "", label: str = "" ) -> None:
        self.data = data
        self._prev =  set(_children)
        self._op = _op
        self.label = label
    
    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    
    def __add__(self, other: Node) -> Node:
        return Node(self.data+ other.data, (self, other), "+")
    
    
    def __mul__(self, other: Node) -> Node:
        return Node(self.data*other.data, (self, other), "*")
    
    

if __name__ ==  "__main__":
    pass