from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """

    # TODO: Implement for Task 1.1.

    my_vals = list(vals)
    my_vals[arg] += epsilon
    print(arg)
    print(*my_vals, my_vals)
    print(*vals, vals)
    delta = f(*my_vals) - f(*vals)
    return delta / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    Visited = []
    output: List[Any] = []

    def get_node(n: Variable) -> None:
        if n.is_constant():  # constant
            return
        if n.unique_id in Visited:  # visited
            return
        if not n.is_leaf():
            for input in n.parents:
                get_node(input)
        Visited.append(n.unique_id)
        output.insert(0, n)

    get_node(variable)
    return output


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    output = topological_sort(variable)

    node_deriv = {}
    node_deriv[variable.unique_id] = deriv
    for n in output:
        if n.is_leaf():  # if node is leaf, we move on to the next down the line
            continue
        if (
            n.unique_id in node_deriv.keys()
        ):  # if node already has a derivative, have to get the current one and add on
            deriv = node_deriv[n.unique_id]
        deriv_tmp = n.chain_rule(
            deriv
        )  # get the chain rule result, deriv is the d_output
        for key, item in deriv_tmp:
            if key.is_leaf():
                key.accumulate_derivative(item)
                continue
            if key.unique_id in node_deriv.keys():
                # decide if already calculated derivative of node
                node_deriv[key.unique_id] += item
            else:
                node_deriv[key.unique_id] = item


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
