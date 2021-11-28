from __future__ import annotations

import inspect as insp
import numpy as np
import pandas as pd

from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Callable, Iterable, List, Optional, Tuple

from .latex import array_to_latex, matrix_to_latex, Brackets


def fill_defaults(method: Callable):
    """
    Helps to avoid writing duplicate code
    in each function with these arguments:

    ```
    precision = precision or self.default_precision
    var_sym = var_sym or self.default_var_sym
    tgt_sym = tgt_sym or self.default_tgt_sym
    ...
    ```
    """
    signature = insp.signature(method).parameters

    _, *signature = signature

    @wraps(method)
    def _wrapper(
        self: Table,
        *args,
        **kwargs,
    ):
        for idx, name in enumerate(signature):
            # Since self argument was skipped
            idx = idx + 1

            # Positional parameter
            if len(args) > idx and args[idx] is None:
                args[idx] = getattr(self, f"default_{name}", None)

            # Keyword parameter
            elif kwargs.get(name) is None:
                kwargs[name] = getattr(self, f"default_{name}", None)

        # Launch method execution
        return method(
            self,
            *args,
            **kwargs,
        )

    return _wrapper


@dataclass(frozen=True)
class Vec:
    data: List

    def __le__(self, oth: float):
        return type(self)([*self.data, oth])

    def __ge__(self, oth: float):
        return type(self)([*self.data, -oth])


class V:
    """
    Allows V[elem1, elem2, elem3] syntax on system definition
    """

    def __getitem__(self, key):
        return Vec(list(key))


V = V()


class Table:
    # Actual system data
    data: np.ndarray

    # Rows & columns mapping
    hlabels: list
    vlabels: list

    # Latex output modifiers
    default_A_sym: str = "A"
    default_b_sym: str = "b"
    default_c_sym: str = "c"
    default_var_sym: str = "x"
    default_tgt_sym: str = "F"
    default_unbound: str = "S_0"
    default_precision: str = 2
    default_brackets: Brackets = Brackets.square

    def __init__(self, c: np.ndarray, A: np.ndarray, b: np.ndarray):
        """Creates simplex table data"""
        constr = np.expand_dims(b, axis=1)
        data = np.hstack((constr, A))
        tgt = np.hstack((0, c))
        self.data = np.vstack((data, tgt))
        labels = list(range(A.shape[1] + A.shape[0]))
        self.hlabels = labels[:A.shape[1]]
        self.vlabels = labels[A.shape[1]:]

    @property
    def A(self):
        """A input matrix"""
        return self.data[:-1, 1:]

    @property
    def b(self):
        """b input array"""
        return self.data[:-1, 0]

    @property
    def c(self):
        """c input array"""
        return self.data[-1, 1:]

    @property
    def F(self):
        """Result value"""
        return self.data[-1, 0]

    @property
    def result(self):
        """Result value alias"""
        return self.F

    @classmethod
    def create(
        cls,
        target: Iterable[float],
        *system: List[Vec],
    ) -> Table:
        """
        Place for logic & validation
        To be used with Vec objects
        """
        system = [(v.data[:-1], v.data[-1]) for v in system]
        A, b = list(map(list, zip(*system)))

        return cls(
            np.array(target, dtype=np.float64),
            np.array(A, dtype=np.float64),
            np.array(b, dtype=np.float64),
        )

    @fill_defaults
    def md_system(
        self,
        precision: Optional[int] = None,
        var_sym: Optional[str] = None,
    ) -> str:
        """System as latex output"""
        pass

    @fill_defaults
    def md_target(
        self,
        precision: Optional[int] = None,
        var_sym: Optional[str] = None,
        tgt_sym: Optional[str] = None,
    ) -> str:
        """Target vector as latex output"""
        pass

    @fill_defaults
    def md_table(
        self,
        precision: Optional[int] = None,
        var_sym: Optional[str] = None,
        tgt_sym: Optional[str] = None,
    ) -> str:
        """System table as latex output"""
        return pd.DataFrame(
            np.round(self.table, precision),
            index=[f"${value}$" for value in self.v_labels],
            columns=[f"${value}$" for value in self.h_labels],
        ).to_markdown()

    @fill_defaults
    def md_A(
        self,
        precision: Optional[int] = None,
        A_sym: Optional[str] = None,
        brackets: Optional[Brackets] = None,
    ) -> str:
        """A matrix as latex output"""
        return matrix_to_latex(self.A, A_sym, precision, brackets)

    @fill_defaults
    def md_b(
        self,
        precision: Optional[int] = None,
        b_sym: Optional[str] = None,
        brackets: Optional[Brackets] = None,
    ) -> str:
        """b array as latex output"""
        return array_to_latex(self.b, b_sym, precision, brackets)

    @fill_defaults
    def md_variables(
        self,
        var_sym: Optional[str] = None,
        precision: Optional[int] = None,
    ):
        return

    @fill_defaults
    def md_c(
        self,
        precision: Optional[int] = None,
        c_sym: Optional[str] = None,
        brackets: Optional[Brackets] = None,
    ) -> str:
        """c array as latex output"""
        return array_to_latex(self.c, c_sym, precision, brackets)

    def add_constraint(
        self,
        equation: Vec,
    ) -> Table:
        """
        Adds constraint to the system, returns new Table
        To be used with Vec object
        """
        # Copy table with one more line to the system
        return Table(
            self.c.copy(),
            np.vstack((self.A.copy(), equation.data[:-1])),
            np.hstack((self.b.copy(), equation.data[-1])),
        )

    def copy(self) -> Table:
        """Convinience method for preserving table state"""

        return Table(
            self.c.copy(),
            self.A.copy(),
            self.b.copy(),
        )


class State(str, Enum):
    FIXER = "FIXER"
    SOLVER = "SOLVER"


ReturnType = Tuple[
    # Result table
    Table,
    # Itermediate steps:
    # 1. step-result Table,
    # 2. solution state - either FIXER or SOLVER
    # Duplicates result table
    List[Tuple[Table, State]],
    # True - solvable
    # False - not solvable,
    bool,
]


class Simplex:
    @classmethod
    def minimize(self, table: Table) -> ReturnType:
        pass

    @classmethod
    def maximiae(self, table: Table) -> ReturnType:
        pass
