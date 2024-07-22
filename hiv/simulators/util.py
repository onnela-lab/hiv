import networkx as nx


class Constraint:
    def is_valid(self, value) -> bool:
        raise NotImplementedError


class Interval(Constraint):
    def __init__(self, low, high, type=None) -> None:
        assert low is None or high is None or low < high
        self.low = low
        self.high = high
        self.type = type

    def is_valid(self, value) -> bool:
        return (
            (self.low is None or self.low <= value)
            and (self.high is None or value <= self.high)
            and (self.type is None or isinstance(value, self.type))
        )


class UnitInterval(Interval):
    def __init__(self) -> None:
        super().__init__(0, 1)


class Simulator:
    arg_constraints: dict[str, Constraint] = {}

    def init(self) -> nx.Graph:
        raise NotImplementedError

    def step(self, graph: nx.Graph) -> nx.Graph:
        raise NotImplementedError

    def run(self, graph: nx.Graph, num_steps: int) -> nx.Graph:
        for _ in range(num_steps):
            graph = self.step(graph)
        return graph

    def validate_args(self) -> None:
        for arg, constraint in self.arg_constraints.items():
            if not constraint.is_valid(getattr(self, arg)):
                raise ValueError(
                    f"Argument `{arg}` does not satisfy constraint `{constraint}`."
                )
