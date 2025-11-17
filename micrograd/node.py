from __future__ import annotations
from math import exp


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
        return f"Value(data={self.data})"

    def __add__(self, other: Node) -> Node:
        """
        Overload the addition operator to create a new node.

        Creates a new node representing the sum of this node and another node.
        The new node's children are set to (self, other) and its operation is "+".
        Sets up the backward propagation function to compute gradients for both
        child nodes using the chain rule (gradient flows equally to both inputs).

        Args:
            other: The other node to add to this node.

        Returns:
            A new Node representing the sum of self and other, with backward
            propagation configured to propagate gradients to both inputs.
        """
        res = Node(self.data + other.data, (self, other), "+")

        def _backward():
            self._grad = 1 * res._grad
            other._grad = 1 * res._grad

        self._backward = _backward

        return res

    def __mul__(self, other: Node) -> Node:
        """
        Overload the multiplication operator to create a new node.

        Creates a new node representing the product of this node and another node.
        The new node's children are set to (self, other) and its operation is "*".
        Sets up the backward propagation function to compute gradients for both
        child nodes using the product rule (d(xy)/dx = y, d(xy)/dy = x).

        Args:
            other: The other node to multiply with this node.

        Returns:
            A new Node representing the product of self and other, with backward
            propagation configured to propagate gradients using the product rule.
        """
        res = Node(self.data * other.data, (self, other), "*")

        def _backward():
            self._grad = other.data * res._grad
            other._grad = self.data * res._grad

        self._backward = _backward

        return res

    def tanh(self) -> Node:
        """
        Compute the hyperbolic tangent of this node.

        Creates a new node representing the tanh activation function applied
        to this node's data value. The tanh function maps values to the range
        (-1, 1) and is commonly used as an activation function in neural networks.
        Sets up the backward propagation function to compute the gradient using
        the derivative of tanh: d(tanh(x))/dx = 1 - tanh²(x).

        Returns:
            A new Node representing tanh(self.data) with this node as its child,
            with backward propagation configured to compute gradients using the
            tanh derivative formula.
        """
        val = exp(2 * self.data) - 1 / exp(2 * self.data) + 1
        res = Node(val, (self,), _op="tanh")

        def _backward():
            self._grad = (1 - val**2) * res._grad

        res._backward = _backward

        return res


if __name__ == "__main__":
    pass
