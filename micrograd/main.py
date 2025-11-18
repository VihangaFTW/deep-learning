from micrograd.node import Node


def main() -> None:
    # ? inputs x1, x2
    x1 = Node(2.0, label="x1")
    x2 = Node(0.0, label="x2")

    # ? weights w1, w2
    w1 = Node(-3.0, label="w1")
    w2 = Node(1.0, label="w2")

    # ? bias of the neuron
    b = Node(6.8813735870195432, label="b")

    # x1*w1 + x2*w2 + b
    x1w1 = x1 * w1
    x1w1.label = "x1*w1"
    x2w2 = x2 * w2
    x2w2.label = "x2*w2"
    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = "x1*w1 + x2*w2"
    n = x1w1x2w2 + b
    n.label = "n"
    o = n.tanh(True)
    o.label = "o"

    print(o)

    # Populate gradients through backpropagation with visualization.
    o.backpropagate(visualize=True)


if __name__ == "__main__":
    main()
