import torch


def autodiff_example():
    """PyTorch example demonstrating the same computation as autodiff.

    This example performs the same forward and backward pass as the autodiff
    implementation, but using PyTorch's automatic differentiation system.
    """
    input1 = torch.Tensor([2.0]).double()
    input1.requires_grad = True

    input2 = torch.Tensor([0.0]).double()
    input2.requires_grad = True

    weight1 = torch.Tensor([-3.0]).double()
    weight1.requires_grad = True

    weight2 = torch.Tensor([1.0]).double()
    weight2.requires_grad = True

    bias = torch.Tensor([6.8813735870195432]).double()
    bias.requires_grad = True

    dot = input1 * weight1 + input2 * weight2 + bias

    output = torch.tanh(dot)

    print("output value: ", output.data.item())
    output.backward()

    # Assert that gradients are computed (they will be after backward()).
    assert input1.grad is not None
    assert input2.grad is not None
    assert weight1.grad is not None
    assert weight2.grad is not None
    assert bias.grad is not None

    print("\nLeaf node gradients after backpropagation:")
    print(f"input1: {input1.grad.data.item()}")
    print(f"input2: {input2.grad.data.item()}")
    print(f"weight1: {weight1.grad.data.item()}")
    print(f"weight2: {weight2.grad.data.item()}")
    print(f"bias: {bias.grad.data.item()}")


if __name__ == "__main__":
    autodiff_example()
