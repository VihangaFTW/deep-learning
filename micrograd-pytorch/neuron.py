from random import uniform
from micrograd.node import Node


class Neuron:
    """A single neuron that computes a weighted sum of inputs with a bias and applies tanh activation."""

    def __init__(self, num_inputs: int) -> None:
        """Initialize a neuron with random weights and bias.

        Args:
            num_inputs: Number of input connections to this neuron.
        """
        # Initialize weights as Node objects with random values between -1 and 1.
        self.weights = [Node(uniform(-1, 1)) for _ in range(num_inputs)]
        # Initialize bias as a Node object with a random value between -1 and 1.
        self.bias = Node(uniform(-1, 1))

    def __call__(self, inputs: list[Node]) -> Node:
        """Forward pass: compute weighted sum of inputs plus bias, then apply tanh activation.

        Args:
            inputs: List of input nodes to the neuron.

        Returns:
            A Node representing the activated output of the neuron.
        """
        # Compute the weighted sum: sum(weights * inputs) + bias.
        weighted_sum: Node = sum(
            (wi * xi for wi, xi in zip(self.weights, inputs)), self.bias
        )
        # Apply tanh activation function to the weighted sum.
        activated_node = weighted_sum.tanh()
        return activated_node


class Layer:
    """A layer of neurons that processes inputs and produces outputs."""

    def __init__(self, inputs_per_neuron: int, num_outputs: int) -> None:
        """Initialize a layer with multiple neurons.

        Args:
            inputs_per_neuron: Number of inputs each neuron in the layer receives.
            num_outputs: Number of outputs produced in this layer. This is identical
                        to the number of neurons in this layer.
        """
        # Create a list of neurons, each receiving num_inputs inputs.
        self.neurons = [Neuron(inputs_per_neuron) for _ in range(num_outputs)]

    def __call__(self, inputs: list[Node]) -> list[Node] | Node:
        """Forward pass: process inputs through all neurons in the layer.

        Args:
            inputs: List of input nodes to the layer.

        Returns:
            A single Node if the layer has one neuron, or a list of Node objects
            if the layer has multiple neurons.
        """
        # Compute output from each neuron in the layer.
        outputs = [neuron(inputs) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs


class MLP:
    """Multi-Layer Perceptron: a feedforward neural network with multiple layers."""

    def __init__(self, inputs_per_neuron: int, neurons_per_layer: list[int]) -> None:
        """Initialize an MLP with the specified architecture.

        Args:
            inputs_per_neuron: Number of inputs to the first layer.
            neurons_per_layer: List specifying the number of neurons in each hidden/output layer.
                              For example, [4, 4, 1] creates a network with 3 layers:
                              first layer has 4 neurons, second has 4, third has 1.
        """
        # Build the layer sizes: [input_size, layer1_size, layer2_size, ...]
        layer_sizes = [inputs_per_neuron] + neurons_per_layer

        # Create layers where each layer i takes layer_sizes[i] inputs and produces layer_sizes[i+1] outputs.
        self.layers = [
            Layer(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(neurons_per_layer))
        ]

    def __call__(self, inputs: list[Node]) -> Node | list[Node]:
        """Forward pass: propagate inputs through all layers of the network.

        Args:
            inputs: List of input nodes to the first layer.

        Returns:
            A single Node if the last layer has one neuron, or a list of Node objects
            if the last layer has multiple neurons.
        """
        # Propagate through each layer sequentially.
        outputs: list[Node] = inputs
        for layer in self.layers:
            # Get Node outputs from the current layer.
            layer_output = layer(outputs)
            # Ensure outputs is always a list for the next layer.
            outputs = layer_output if isinstance(layer_output, list) else [layer_output]
        # Return the final layer's output node(s).
        # If the last layer has one neuron, return a single Node; otherwise return the list.
        return outputs[0] if len(outputs) == 1 else outputs


if __name__ == "__main__":
    # Convert input values to Node objects for consistency.
    inputs = [Node(2.0), Node(3.0), Node(-1.0)]
    n = MLP(3, [4, 4, 1])
    output = n(inputs)
    # Since the last layer has 1 neuron, output is a single Node.
    assert isinstance(output, Node), "Expected single Node output"

    # perform gradient backpropagation and visualize results
    output.backpropagate(True)
