from __future__ import annotations

import inspect as insp
import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from itertools import product
from functools import wraps
from typing import Any, Callable, Iterable, List, Optional

from .latex import array_to_latex, matrix_to_latex, equation_system, Brackets


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
        args = list(args)

        for idx, name in enumerate(signature):
            # Since self argument was skipped
            idx = idx + 1

            # Positional parameter
            if len(args) > idx and args[idx] is None:
                args[idx] = getattr(self, f"default_{name}", None)

            # Keyword parameter
            elif idx > len(args) and kwargs.get(name, None) is None:
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
    inverse: bool = False

    def __le__(self, oth: float):
        return type(self)([*self.data, -oth if self.inverse else oth])

    def __ge__(self, oth: float):
        return type(self)([*self.data, oth if self.inverse else -oth])

    @property
    def inv(self):
        self.data = [-elem for elem in self.data]

        self.inverse = not self.inverse

        return self


class V:
    """
    Allows V[elem1, elem2, ..., elemN] syntax on system definition
    """

    def __getitem__(self, key):
        return Vec(list(key))


V = V()


def pretty_coefficient(value: np.float64, precision: int) -> str:
    value = np.round(value, precision)

    if np.float64.is_integer(value):
        value = value.astype(int)

    return f"{'-' if value < 0 else ''}{'' if (val := np.abs(value)) == 1 else val.astype(str)}"


def pretty_value(value: np.float64, precision: int) -> str:
    value = np.round(value, precision)

    if np.float64.is_integer(value):
        value = value.astype(int)

    return value.astype(str)


def pretty_sum(array: Iterable[str]):
    return "+".join(array).replace("+-", "-")


@dataclass
class Format:
    # Source table reference
    victim: Table

    # Markdown output modifiers
    default_A_sym: str = "A"
    default_b_sym: str = "b"
    default_c_sym: str = "c"
    default_var_sym: str = "x"
    default_tgt_sym: str = "F"
    default_unbound: str = "s_0"
    default_precision: str = 2
    default_brackets: Brackets = Brackets.square

    @fill_defaults
    def system(
        self,
        precision: Optional[int] = None,
        var_sym: Optional[str] = None,
    ) -> str:
        """System as latex output"""
        # Create matrix (num of constraints X total number of variables)
        sq = np.zeros((len(self.victim.vlabels), self.victim.var_num))

        # Go along 0 axis of A
        for row, vec in enumerate(self.victim.A):
            # Variable is not present - set as +-1 depending on the constraint
            sq[row, self.victim.vlabels[row] - 1] = -1 if self.victim.b[row] < 0 else 1

            # Go along 1 axis of A
            for elem, col in zip(vec, self.victim.hlabels):
                # Set respective variables
                sq[row, col - 1] = elem

        # Constraints are positive
        constr = np.abs(self.victim.b)
        # Labels are sorted in ascention order
        labels = [f"{var_sym}_{idx + 1}" for idx in range(self.victim.var_num)]

        # Build equation system
        return equation_system(
            sq,
            constr,
            labels,
            precision,
        )

    @fill_defaults
    def target(
        self,
        precision: Optional[int] = None,
        var_sym: Optional[str] = None,
        tgt_sym: Optional[str] = None,
    ) -> str:
        """Target vector as latex output"""
        # Cast c to pretty strings
        c = [pretty_coefficient(elem, precision) for elem in self.victim.c]
        # Make equation
        eq = pretty_sum(
            [f"{elem}{var_sym}_{label}" for elem, label in zip(c, self.victim.hlabels)]
        )

        # Return target equation
        return f"$${tgt_sym} = {eq} \\rightarrow {'min' if self.victim.minimize else 'max'}$$"

    @fill_defaults
    def table(
        self,
        precision: Optional[int] = None,
        var_sym: Optional[str] = None,
        tgt_sym: Optional[str] = None,
        unbound: Optional[str] = None,
    ) -> str:
        """System table as latex output"""
        return pd.DataFrame(
            np.round(self.victim.data, precision),
            index=[f"${var_sym}_{value}$" for value in self.victim.vlabels]
            + [f"${tgt_sym}$"],
            columns=[f"${unbound}$"]
            + [f"${var_sym}_{value}$" for value in self.victim.hlabels],
        ).to_markdown()

    @fill_defaults
    def A(
        self,
        precision: Optional[int] = None,
        A_sym: Optional[str] = None,
        brackets: Optional[Brackets] = None,
    ) -> str:
        """A matrix as latex output"""
        return matrix_to_latex(self.victim.A, A_sym, precision, brackets)

    @fill_defaults
    def b(
        self,
        precision: Optional[int] = None,
        b_sym: Optional[str] = None,
        brackets: Optional[Brackets] = None,
    ) -> str:
        """b array as latex output"""
        return array_to_latex(self.victim.b, b_sym, precision, brackets)

    @fill_defaults
    def c(
        self,
        precision: Optional[int] = None,
        c_sym: Optional[str] = None,
        brackets: Optional[Brackets] = None,
    ) -> str:
        """c array as latex output"""
        return array_to_latex(self.victim.c, c_sym, precision, brackets)

    @fill_defaults
    def var_zero_constraint(self, var_sym: Optional[str] = None) -> str:
        """Variables with greater-or-equal constraint"""
        # lower-to-greater ordered variables
        vars = [f"{var_sym}_{idx}" for idx in range(self.victim.var_num)]
        # With >= 0 constraint
        res = ", ".join(vars) + " ≥ 0"
        return f"$${res}$$"

    @fill_defaults
    def base_vars(
        self,
        var_sym: Optional[str] = None,
        precision: Optional[int] = None,
    ) -> str:
        b = [pretty_value(elem, precision) for elem in self.victim.b]

        eq = ", ".join(
            f"{var_sym}_{label} = {value}"
            for value, label in zip(b, self.victim.vlabels)
        )

        return f"$${eq}$$"

    @fill_defaults
    def free_vars(
        self,
        var_sym: Optional[str] = None,
    ) -> str:
        eq = " = ".join(f"{var_sym}_{label}" for label in self.victim.hlabels)

        return f"$${eq} = 0$$"

    @fill_defaults
    def solution(
        self,
        var_sym: Optional[str] = None,
        precision: Optional[int] = None,
    ) -> str:
        eq = ", ".join(
            f"{var_sym}_{label} = {pretty_value(value, precision)}"
            for value, label in zip(self.victim.real_b, self.victim.rlabels)
        )

        return f"$${eq}$$"

    @fill_defaults
    def check(
        self,
        against: np.ndarray,
        zeros: bool = False,
        var_sym: Optional[str] = None,
        tgt_sym: Optional[str] = None,
        precision: Optional[str] = None,
    ) -> str:
        # Create string representation of real labels
        labels = [f"{var_sym}_{label}" for label in self.victim.rlabels]
        scoeffs = [pretty_coefficient(value, precision) for value in self.victim.real_b]
        sscoeffs = [pretty_value(value, precision) for value in self.victim.real_b]
        sagainst = [pretty_value(value, precision) for value in against]

        # Make line with coefficients
        fline = pretty_sum(
            f"{coef}{label}"
            for coef, label, rcoef in zip(scoeffs, labels, self.victim.real_b)
            if not zeros or rcoef != 0
        )

        # Make line with sum
        sline = pretty_sum(
            f"{coef}\cdot{f'({value})' if rvalue < 0 else value}"
            for coef, value, rcoef, rvalue in zip(
                sscoeffs, sagainst, self.victim.real_b, against
            )
            if not zeros or (rcoef != 0 and rvalue != 0)
        )

        # Compute result array and cast it to strings
        result = self.victim.real_b * against
        sresult = [
            pretty_value(value, precision)
            for value in result
            if not zeros or value != 0
        ]
        tline = pretty_sum(sresult)

        rresult = self.victim.result * (-1 if self.victim.minimize else 1)
        return f"$${tgt_sym} = {pretty_value(rresult, precision)} = {fline} = {sline} = {tline} = {pretty_value(np.sum(result), precision)}$$"


class Table:
    # Actual system data
    data: np.ndarray
    minimize: bool = True

    # Rows & columns mapping
    hlabels: list
    vlabels: list

    # "Real" labels
    rlabels: list

    def __init__(
        self,
        c: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        vlabels: List[int],
        hlabels: List[str],
        rlabels: Optional[List[str]] = None,
        F: np.float64 = 0.0,
        minimize: bool = True,
    ):
        """Creates simplex table data"""
        # Create simplex table
        constr = np.expand_dims(b, axis=1)
        data = np.hstack((constr, A))
        tgt = np.hstack((F, c))

        # Set result table
        self.data = np.vstack((data, tgt))

        # Set labels
        self.vlabels = vlabels
        self.hlabels = hlabels
        self.rlabels = rlabels or list(hlabels)

        # Set opt target
        self.minimize = minimize

    @property
    def A(self) -> np.ndarray:
        """A input matrix"""
        return self.data[:-1, 1:]

    @property
    def b(self) -> np.ndarray:
        """b input array"""
        return self.data[:-1, 0]

    @property
    def c(self) -> np.ndarray:
        """c input array"""
        return self.data[-1, 1:]

    @property
    def s0(self) -> np.ndarray:
        """
        s0 column with free coefficients
        (idk if this is the way to call s0)
        """
        return self.b

    @property
    def F(self) -> np.ndarray:
        """Result value"""
        return self.data[-1, 0]

    @property
    def result(self) -> np.ndarray:
        """Result value alias"""
        return self.F

    @property
    def real_b(self) -> np.ndarray:
        """values of source variables"""
        current = {label: value for value, label in zip(self.b, self.vlabels)}
        values = [current.get(label, np.float64(0)) for label in self.rlabels]

        return np.array(values)

    @property
    def var_num(self) -> int:
        """Number of bound/unbound variables"""
        return len(self.hlabels) + len(self.vlabels)

    @classmethod
    def straight(
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
        labels = list(range(1, len(A[0]) + len(A) + 1))

        return cls(
            np.array(target, dtype=np.float64),
            np.array(A, dtype=np.float64),
            np.array(b, dtype=np.float64),
            labels[len(A[0]) :],
            labels[: len(A[0])],
        )

    @classmethod
    def inverse(
        cls,
        target: Iterable[float],
        *system: List[Vec],
    ) -> Table:
        system = [(v.data[:-1], v.data[-1]) for v in system]
        A, b = list(map(list, zip(*system)))
        labels = list(range(1, len(A[0]) + len(A) + 1))

        return cls(
            np.array(b, dtype=np.float64).T * -1,
            np.array(A, dtype=np.float64).T * -1,
            np.array(target, dtype=np.float64).T * -1,
            labels[len(A[0]) :],
            labels[: len(A[0])],
        )

    def __rshift__(self, oth: Callable):
        """
        Allows to set optimization target in functional style:
        ```
        Table >> (min | max)
        ```
        """
        # If -> min
        if oth is min:
            # Check if target is not min, otherwise leave F as is
            if not self.minimize:
                self.data[-1, :] *= -1

                self.minimize = True

            return self

        # If -> max
        if oth is max:
            # Check if target is min, otherwise leave F as is
            if self.minimize:
                # multiply F vector by -1
                self.data[-1, :] *= -1

                self.minimize = False

            return self

        # Other values are not allowed
        raise ValueError("Only `min` & `max` funtions are allowed for `>` use")

    def __getitem__(self, item: Any) -> np.ndarray:
        # Forward __getitem__ to underlying table
        return self.data.__getitem__(item)

    def __setitem__(self, key: Any, value: Any) -> None:
        # Forward __setitem__ to underlying table
        return self.data.__setitem__(key, value)

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
            self.vlabels + [self.var_num + 1],
            self.hlabels,
            self.rlabels,
            self.F,
            self.minimize,
        )

    @property
    def clone(self) -> Table:
        """Convinience method for preserving table state"""

        return Table(
            self.c.copy(),
            self.A.copy(),
            self.b.copy(),
            list(self.vlabels),
            list(self.hlabels),
            list(self.rlabels),
            self.F,
            self.minimize,
        )


@dataclass
class SimplexResult:
    fixed: bool = field(default=True)
    solved: bool = field(default=False)
    fix_history: list = field(default_factory=list)
    sol_history: list = field(default_factory=list)

    def __bool__(self):
        return self.fixed and self.solved

    @property
    def history(self):
        return self.fix_history + self.sol_history[1:]

    @property
    def source(self):
        if not self.fix_history:
            raise ValueError("Incomplete result")

        return self.fix_history[0]

    @property
    def result(self):
        if not self.sol_history:
            raise ValueError("Incomplete result")

        return self.sol_history[-1]


class Simplex:
    @classmethod
    def _find_negative(cls, table: Table, idx: int) -> Optional[int]:
        # Values of non-basis variables in row of negative s0
        row = table[idx, 1:].flatten()

        # Look for first negative
        idx = np.argmin(row)

        # Extract index (compensate for missing leading value by adding one to index)
        return (idx + 1) if row[idx] < 0 else None

    @classmethod
    def _find_pivot(cls, table: Table) -> Optional[int]:
        # Values of non-basis variables in F-row (w/o s0-column value)
        row = table.c.flatten()

        # Look for first positive/negative
        idx = (np.argmax if table.minimize else np.argmin)(row)

        # Extract index (compensate for missing leading value by adding one to index)
        return (idx + 1) if (row[idx] > 0 if table.minimize else row[idx] < 0) else None

    @classmethod
    def _find_minimal_ratio(
        cls,
        table: Table,
        jdx: int,
    ) -> Optional[int]:
        # s0 & solver columns
        column = table[:-1, 0].flatten()
        solver_column = table[:-1, jdx].flatten()

        # I'll map division results to array of +np.inf
        # so any values, which either break divison (or division result < 0)
        # will end up as +np.inf and do not mess up argmin
        infs = np.full_like(column, +np.inf)
        idx = np.argmin(
            np.divide(
                column,
                solver_column,
                out=infs,
                where=column * solver_column > 0,
            )
        )

        # Check if there were any valid ratios & return idx
        return idx if infs[idx] != +np.inf else None

    @classmethod
    def _find_fixer(cls, table: Table) -> tuple[tuple[int, int], bool]:
        # Values of s0-column (w/o F-row value)
        column = table[:-1, 0].flatten()

        # Look for first negative
        indices = np.flatnonzero(column < 0)

        # Extract index
        idx = indices[0] if len(indices) else None

        if idx is None:
            return None, True

        jdx = cls._find_negative(table, idx)

        # No column - either solved, or not solvable at all
        if jdx is None:
            return None, False

        # Find minimal positive ratio of solver s0 & solver column
        idx = cls._find_minimal_ratio(table, jdx) if jdx else None

        # There is no solver row
        if idx is None:
            return None, False

        # Return tuple if both of indices are valid
        return (idx, jdx), True

    @classmethod
    def _find_solver(cls, table: Table) -> tuple[tuple[int, int], bool]:
        jdx = cls._find_pivot(table)

        # No column - solved
        if jdx is None:
            return None, True

        # Find minimal positive ratio of solver s0 & solver column
        idx = cls._find_minimal_ratio(table, jdx) if jdx else None

        # There is no solver row
        if idx is None:
            return None, False

        # Return tuple if both of indices are valid
        return (idx, jdx), True

    @classmethod
    def _step(cls, table: Table, idx: int, jdx: int):
        # Table dimensions
        x, y = table.data.shape

        # Adjust labels. Sub 1 from jdx since labels do not include s0
        table.vlabels[idx], table.hlabels[jdx - 1] = (
            table.hlabels[jdx - 1],
            table.vlabels[idx],
        )

        # Solver-values shortcut
        solver = table[idx, jdx]

        # Flatten makes copies
        solver_row = table[idx, :].flatten()
        solver_column = table[:, jdx].flatten()

        # Is there any other way?
        for i, j in product(range(x), range(y)):
            if i == idx and j == jdx:
                table[i, j] = 1 / solver
            elif i == idx:
                table[i, j] /= solver
            elif j == jdx:
                table[i, j] /= -solver
            else:
                table[i, j] -= solver_column[i] * solver_row[j] / solver

        return table

    @classmethod
    def fix(cls, table: Table):
        # First history entry is always source table
        history = [table]
        while fixer := cls._find_fixer(table):
            # Dispatch _find_fixer result
            pos, fixable = fixer

            # Table wont fix
            if not fixable:
                fixed = False

                break

            # Fixable & no fix position - fixed
            if not pos:
                fixed = True

                break

            # Perform matrix transofrm
            table = cls._step(table.clone, *pos)

            # Write resulting table to history
            history.append(table)

        return history, fixed

    @classmethod
    def solve(cls, table: Table):
        # First history entry is always source table
        history = [table]
        while solver := cls._find_solver(table):
            # Dispatch _find_solver result
            pos, solvable = solver

            # Table wont solve
            if not solvable:
                solved = False

                break

            # Solvable & no solve position - solved
            if not pos:
                solved = True

                break

            # Perform matrix transofrm
            table = cls._step(table.clone, *pos)

            # Write resulting table to history
            history.append(table)

        return history, solved

    @classmethod
    def resolve(cls, table: Table) -> SimplexResult:
        # Trying to fix input table
        fix_hist, fixed = cls.fix(table)

        # Unable to fix input table
        if not fixed:
            return SimplexResult(fixed, False, fix_hist)

        # The last table is resulting fix-table
        *_, table = fix_hist

        # Tryin to solve fixed table
        sol_hist, solved = cls.solve(table)

        return SimplexResult(fixed, solved, fix_hist, sol_hist)
