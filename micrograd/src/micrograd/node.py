from __future__ import annotations
from math import exp, tanh
from pathlib import Path


class Node:
    """
    Represents a node in a computational graph for automatic differentiation.

    Each node stores a numerical value (data) and maintains connections to its
    parent nodes (children in the forward pass). This enables both forward
    computation and backward propagation of gradients. Each operation node
    defines a _backward function that computes and accumulates gradients
    during the backward pass.
    """

    def __init__(
        self,
        data: float,
        _children: tuple[Node, ...] = (),
        _op: str = "",
        label: str = "",
        _grad: float = 0.0,
    ) -> None:
        """
        Initialize a Node in the computational graph.

        Args:
            data: The numerical value stored in this node.
            _children: Tuple of child nodes (inputs to this node's operation).
                       Stored as a set internally for fast lookup and deduplication.
            _op: The operation that produced this node (e.g., "+", "*", "" for leaf nodes).
            label: Human-readable label for visualization purposes.
            _grad: The gradient of this node (used during backpropagation).
            _backward: Function that computes gradients for child nodes during
                      backward propagation. Defaults to a no-op lambda for leaf nodes.
        """
        self.data = data
        # Convert children tuple to set for O(1) membership testing and deduplication.
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._grad = _grad
        self._backward = lambda: None

    def __repr__(self) -> str:
        """
        String representation of the node.

        Returns:
            A string showing the node's data value.
        """
        return f"Node(data={self.data})"

    def __add__(self, other: Node | float) -> Node:
        """
        Overload the addition operator to create a new node.

        Creates a new node representing the sum of this node and another node or float.
        If other is a float, it is automatically converted to a Node. The new node's
        children are set to (self, other) and its operation is "+". Sets up the backward
        propagation function to compute gradients for both child nodes using the chain
        rule (gradient flows equally to both inputs).

        Args:
            other: The other node or float to add to this node.

        Returns:
            A new Node representing the sum of self and other, with backward
            propagation configured to propagate gradients to both inputs.
        """
        other = other if isinstance(other, Node) else Node(other)

        res = Node(self.data + other.data, (self, other), "+")

        def _backward():
            # Accumulate gradients using += to handle nodes used in multiple operations.
            self._grad += 1 * res._grad
            other._grad += 1 * res._grad

        res._backward = _backward

        return res

    def __radd__(self, other: Node | float) -> Node:
        """
        Overload the right addition operator to handle number + Node cases.

        Enables addition when a number appears on the left side of a Node
        (e.g., 2 + node). Delegates to __add__ to handle the actual computation.

        Args:
            other: The number or Node to add to this node.

        Returns:
            A new Node representing the sum of other and self.
        """
        return self + other

    def __rmul__(self, other: Node | float) -> Node:
        """
        Overload the right multiplication operator to handle number * Node cases.

        Enables multiplication when a number appears on the left side of a Node
        (e.g., 2 * node). Delegates to __mul__ to handle the actual computation.

        Args:
            other: The number or Node to multiply with this node.

        Returns:
            A new Node representing the product of other and self.
        """
        return self * other

    def __mul__(self, other: Node | float) -> Node:
        """
        Overload the multiplication operator to create a new node.

        Creates a new node representing the product of this node and another node or float.
        If other is a float, it is automatically converted to a Node. The new node's
        children are set to (self, other) and its operation is "*". Sets up the backward
        propagation function to compute gradients for both child nodes using the product
        rule (d(xy)/dx = y, d(xy)/dy = x).

        Args:
            other: The other node or float to multiply with this node.

        Returns:
            A new Node representing the product of self and other, with backward
            propagation configured to propagate gradients using the product rule.
        """
        other = other if isinstance(other, Node) else Node(other)

        res = Node(self.data * other.data, (self, other), "*")

        def _backward():
            # Accumulate gradients using += to handle nodes used in multiple operations.
            self._grad += other.data * res._grad
            other._grad += self.data * res._grad

        res._backward = _backward

        return res

    def __pow__(self, other: float | int) -> Node:
        """
        Overload the power operator to raise this node to a constant power.

        Creates a new node representing this node raised to the power of other.
        The new node's child is set to (self,) and its operation is "**{other}".
        Sets up the backward propagation function using the power rule:
        d(x^n)/dx = n * x^(n-1).

        Args:
            other: The exponent (must be an int or float).

        Returns:
            A new Node representing self raised to the power of other, with backward
            propagation configured using the power rule.

        Raises:
            AssertionError: If other is not an int or float.
        """
        assert isinstance(other, (int, float))

        res = Node(self.data**other, (self,), f"**{other}")

        def _backward():
            # Accumulate gradients using += to handle nodes used in multiple operations.
            # Power rule: d(x^n)/dx = n * x^(n-1)
            self._grad += other * (self.data ** (other - 1)) * res._grad

        res._backward = _backward

        return res

    def __truediv__(self, other: Node) -> Node:
        """
        Overload the division operator to divide this node by another node.

        Implements division as multiplication by the reciprocal: self / other = self * other^(-1).
        This reuses the multiplication and power operations, ensuring consistent
        backward propagation behavior.

        Args:
            other: The node to divide by (divisor).

        Returns:
            A new Node representing self divided by other.
        """
        return self * other**-1

    def exp(self) -> Node:
        """
        Compute the exponential (e^x) of this node.

        Creates a new node representing e raised to the power of this node's data value.
        The new node's child is set to (self,) and its operation is "exp". Sets up the
        backward propagation function using the derivative of exp: d(e^x)/dx = e^x.

        Returns:
            A new Node representing exp(self.data) with this node as its child,
            with backward propagation configured using the exponential derivative.
        """
        res = Node(exp(self.data), (self,), "exp")

        def _backward():
            # Accumulate gradients using += to handle nodes used in multiple operations.
            # Derivative of exp: d(e^x)/dx = e^x
            self._grad += res.data * res._grad

        res._backward = _backward

        return res

    def __neg__(self) -> Node:
        """
        Overload the unary negation operator to negate this node.

        Implements negation as multiplication by -1: -self = self * (-1).
        This reuses the multiplication operation, ensuring consistent backward
        propagation behavior.

        Returns:
            A new Node representing the negation of self.
        """
        return self * -1

    def __sub__(self, other: Node | float) -> Node:
        """
        Overload the subtraction operator to subtract another node or float from this node.

        Implements subtraction as addition of the negated value: self - other = self + (-other).
        If other is a float, it is automatically converted to a Node through the negation
        and addition operations. This reuses the addition and negation operations, ensuring
        consistent backward propagation behavior.

        Args:
            other: The node or float to subtract from this node.

        Returns:
            A new Node representing self minus other.
        """
        return self + (-other)

    def __rsub__(self, other: Node | float) -> Node:
        """
        Overload the right subtraction operator to handle number - Node cases.

        Enables subtraction when a number appears on the left side of a Node
        (e.g., 5 - node). Computes other - self using addition with negation:
        other - self = (-self) + other. The __add__ method handles converting
        other to a Node if it's a float.

        Args:
            other: The number or Node to subtract from.

        Returns:
            A new Node representing other minus self.
        """
        return (-self) + other

    def tanh(self, exp: bool = False) -> Node:
        """
        Compute the hyperbolic tangent of this node.

        Creates a new node representing the tanh activation function applied
        to this node's data value. The tanh function maps values to the range
        (-1, 1) and is commonly used as an activation function in neural networks.
        Sets up the backward propagation function to compute the gradient using
        the derivative of tanh: d(tanh(x))/dx = 1 - tanh²(x).

        The computation can be performed in two ways:
        - If exp=False (default): Uses the math.tanh() function directly for efficiency.
        - If exp=True: Computes tanh via exponential operations: tanh(x) = (e^(2x) - 1) / (e^(2x) + 1).

        Args:
            exp: If True, computes tanh using exponential operations instead of the direct
                 math.tanh() function. Defaults to False.

        Returns:
            A new Node representing tanh(self.data) with this node as its child,
            with backward propagation configured to compute gradients using the
            tanh derivative formula.
        """
        if exp:
            e = (2 * self).exp()
            tanh_via_exp = (e - 1) / (e + 1)
            val = tanh_via_exp.data
        else:
            val = tanh(self.data)

        res = Node(val, (self,), _op="tanh")

        def _backward():
            # Accumulate gradients using += to handle nodes used in multiple operations.
            # Derivative of tanh: d(tanh(x))/dx = 1 - tanh²(x)
            self._grad += (1 - val**2) * res._grad

        res._backward = _backward

        return res

    def _topological_sort(self) -> list["Node"]:
        """
        Performs a topological sort of the computational graph using depth-first search.

        Traverses the graph starting from this node and builds a topological ordering
        where each node appears before its parent nodes. This ordering is useful for
        backward propagation, ensuring gradients are computed in the correct order
        (from output to inputs).

        Returns:
            A list of nodes in topological order, where each node appears before
            its parent nodes. This ordering ensures that when traversing the list
            in reverse, gradients can be computed correctly during backward propagation.
        """
        topo_ordering: list[Node] = []

        # Set for O(1) membership lookup.
        visited: set[Node] = set()

        def build_topo(parent_node: Node):
            """
            Recursively builds the topological ordering using DFS.

            Visits each node once, recursively processes all child nodes first,
            then appends the current node to the ordering. This ensures a valid
            topological order where dependencies (children) come before dependents (parents).
            """
            if parent_node not in visited:
                visited.add(parent_node)
                # Process all children first to ensure they appear before the parent.
                for child_node in parent_node._prev:
                    build_topo(child_node)
                # Append parent after all children have been processed.
                topo_ordering.append(parent_node)

        build_topo(self)

        return topo_ordering

    def _collect_nodes(self) -> set["Node"]:
        """
        Collects all nodes in the computational graph starting from this node.

        Returns:
            A set of all nodes in the graph.
        """
        nodes: set[Node] = set()

        def build(parent_node: Node):
            """Recursively collects all nodes in the graph."""
            if parent_node not in nodes:
                nodes.add(parent_node)
                for child_node in parent_node._prev:
                    build(child_node)

        build(self)
        return nodes

    def backpropagate(self, visualize: bool = False) -> None:
        """
        Performs backward propagation to compute gradients for all nodes.

        This method executes the complete backward propagation algorithm to compute
        gradients for all nodes in the computational graph starting from this node.
        Optionally visualizes the graph before and after backpropagation.

        The backpropagation process:
        1. Reset all gradients to zero for clean runs.
        2. Initialize this node's gradient to 1.0 (base case: d(output)/d(output) = 1).
        3. Get a topological ordering of all nodes (children before parents).
        4. Traverse nodes in reverse topological order (parents before children).
        5. Call each node's _backward() function to propagate gradients to its children.

        Args:
            visualize: If True, visualizes the graph before and after backpropagation
                      using graphviz. Requires graphviz to be installed.
        """
        # Optionally visualize the graph before backpropagation.
        if visualize:
            try:
                from micrograd.graph import make_graph

                # Create outputs directory if it doesn't exist.
                output_dir = Path("outputs")
                output_dir.mkdir(exist_ok=True)

                graph = make_graph(self)
                graph.render(output_dir / "graph-before-backprop", view=True)
            except ImportError:
                print("Warning: graphviz not available, skipping visualization.")

        # Reset all gradients to zero before backpropagation for clean runs.
        nodes = self._collect_nodes()
        for node in nodes:
            node._grad = 0.0

        # Base case: gradient of output node = d(output)/d(output) = 1.0.
        self._grad = 1.0

        # Sort the nodes so that child nodes come before parent nodes.
        # Gradient calculation of a node requires gradients from its parent nodes
        # to be computed first. By reversing the topological order, we ensure
        # parents are processed before their children during backpropagation.
        topo_order = self._topological_sort()

        # Traverse nodes in reverse topological order (from output to inputs).
        # This ensures that when we compute a node's gradient, all parent gradients
        # have already been computed and are available.
        for node in reversed(topo_order):
            # Call the node's backward function to propagate gradients to its children.
            node._backward()

        # Optionally visualize the graph after backpropagation.
        if visualize:
            try:
                from micrograd.graph import make_graph

                # Create outputs directory if it doesn't exist.
                output_dir = Path("outputs")
                output_dir.mkdir(exist_ok=True)

                graph = make_graph(self)
                graph.render(output_dir / "graph-after-backprop", view=True)
            except ImportError:
                print("Warning: graphviz not available, skipping visualization.")


if __name__ == "__main__":
    pass
