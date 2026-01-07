"""Type checking utilities for PyTorch nn.Module.

This module provides boilerplate for fixing the `Any` return type of PyTorch nn.Module
instances when they are called. By using the `apply_module` function, we can preserve
type hints so that the return type is properly inferred as `torch.Tensor` (or other types),
enabling IDE autocomplete and type checking suggestions.

Modified code from: https://github.com/pytorch/pytorch/issues/74746
"""

from typing import Callable, ParamSpec, Protocol, TypeVar
from typing_extensions import Self


P = ParamSpec("P")
R = TypeVar("R", covariant=True)


class _Module(Protocol[P, R]):
    """Protocol allowing us to unwrap `forward`."""

    def forward(self: Self, *args: P.args, **kwargs: P.kwargs) -> R: ...

    def __call__(self: Self, *args: P.args, **kwargs: P.kwargs) -> R: ...


def apply_module(m: _Module[P, R]) -> Callable[P, R]:
    """Returns the provided module unchanged, but with type hints preserved.

    Args:
        m: An instance of a subclass of `torch.nn.Module` to apply.

    Returns:
        m unchanged.
    """
    return m
