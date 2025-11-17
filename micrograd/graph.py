from graphviz import Digraph
from node import Node


def topological_sort(root: Node) -> list[Node]:
    """
    Performs a topological sort of the computational graph using depth-first search.

    Traverses the graph starting from the root node and builds a topological ordering
    where each node appears before its parent nodes. This ordering is useful for
    backward propagation, ensuring gradients are computed in the correct order
    (from output to inputs).

    The algorithm uses a recursive DFS approach: for each node, it first recursively
    processes all child nodes, then appends the current node to the ordering.
    This ensures children appear before parents in the final ordering.

    Args:
        root: The root node of the computational graph to sort.

    Returns:
        A list of nodes in topological order, where each node appears before
        its parent nodes. This ordering ensures that when traversing the list
        in reverse, gradients can be computed correctly during backward propagation.
    """
    topo_ordering: list[Node] = []

    # set for O(1) membership lookup
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

    build_topo(root)

    return topo_ordering


def collect_nodes_and_edges(root: Node) -> tuple[set[Node], set[tuple[Node, Node]]]:
    """
    Traverses the computational graph starting from the root node.

    Collects all nodes and edges in the graph by recursively visiting
    each node and its children. Returns the complete set of nodes and
    edges for visualization purposes.

    Args:
        root: The root node of the computational graph.

    Returns:
        A tuple containing:
        - nodes: Set of all nodes in the graph.
        - edges: Set of tuples representing edges (child_node, parent_node).
    """
    nodes: set[Node] = set()
    edges: set[tuple[Node, Node]] = set()

    def build(parent_node):
        """
        Recursively builds the graph representation.

        Visits each node once, adds it to the nodes set, and creates
        edges from each child node to the current node. Then recursively
        processes all child nodes.
        """
        if parent_node not in nodes:
            nodes.add(parent_node)
            for child_node in parent_node._prev:
                # Edge direction: child -> parent (child_node -> node).
                edges.add((child_node, parent_node))
                build(child_node)

    build(root)
    return nodes, edges


def draw_graph(root: Node) -> Digraph:
    """
    Visualizes the computational graph using Graphviz.

    Creates a directed graph visualization showing all nodes and edges
    in the computational graph starting from the root node.

    Args:
        root: The root node of the computational graph to visualize.

    Returns:
        A Digraph object representing the computational graph.
    """

    # Create a new Digraph object with SVG format and left-to-right ranking.
    graph = Digraph(format="svg", graph_attr={"rankdir": "LR"})

    nodes, edges = collect_nodes_and_edges(root)

    # Create nodes in the graph.
    for node in nodes:
        node_id = str(id(node))
        graph.node(
            name=node_id,
            label=f"{node.label} | data {node.data:.4f} | grad {node._grad:.4f}",
            shape="record",
        )

        # If node has an operation, create an operation node and connect it.
        if node._op:
            op_node_id = node_id + node._op
            graph.node(name=op_node_id, label=node._op)
            graph.edge(op_node_id, node_id)

    # Create edges between nodes.
    for child_node, parent_node in edges:
        child_id = str(id(child_node))
        parent_id = str(id(parent_node))

        # Connect to parent's op node if it exists, otherwise to parent node directly.
        if parent_node._op:
            parent_target = parent_id + parent_node._op
        else:
            parent_target = parent_id

        graph.edge(child_id, parent_target)

    return graph
