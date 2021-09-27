from dataclasses import dataclass
from itertools import product, count
from typing import Optional

import numpy as np
import pandas as pd

from IPython.display import display_markdown


@dataclass
class Table:
    table: np.ndarray
    h_labels: list
    v_labels: list

    @classmethod
    def create(cls, c: np.ndarray, A: np.ndarray, b: np.ndarray):
        # Add F-values row
        table = np.vstack([
            A, np.expand_dims(
                c, axis=0
            )
        ])

        # Add s_0 column with 0 at F row
        table = np.hstack([
            np.expand_dims(
                [*b, 0.], axis=1
            ), table
        ])

        # Create indices & column names
        _, columns = A.shape
        labels = [f"x_{idx}" for idx in range(1, sum(A.shape) + 1)]
        labels = ['s_0', *labels[:columns]], [*labels[columns:], 'F']

        # create helper structure
        return cls(table, *labels)
    
    def find_negative(self, idx: int) -> Optional[int]:
        # Values of non-basis variables in row of negative s0
        row = self.table[idx, 1:].flatten()

        # Look for first negative
        indices = np.flatnonzero(row < 0)

        # Extract index
        return (indices[0] + 1) if len(indices) else None

    def find_positive(self) -> Optional[int]:
        # Values of non-basis variables in F-row (w/o s0-column value)
        row = self.table[-1, 1:].flatten()

        # Look for first positive
        indices = np.flatnonzero(row > 0)

        # Extract index
        return (indices[0] + 1) if len(indices) else None

    def find_minimal_ratio(self, jdx: int) -> Optional[int]:
        # s0 & solver columns
        column = self.table[:-1, 0].flatten()
        solver_column = self.table[:-1, jdx].flatten()

        # I'll map division results to array of +np.inf
        # so any values, which either break divison (or division result < 0)
        # will end up as +np.inf and do not mess up argmin
        pos_inf = np.full_like(column, +np.inf)
        idx = np.argmin(
            np.divide(
                column,
                solver_column,
                out=pos_inf,
                where=column * solver_column > 0
            )
        )

        # Check if there were any valid ratios
        return idx if pos_inf[idx] != +np.inf else None

    def find_solver(self) -> tuple[int, int]:
        # Values of s0-column (w/o F-row value)
        column = self.table[:-1, 0].flatten()

        # Index of first negative in s0
        # filter enumerate pairs for correct idx
        indices = np.flatnonzero(column < 0)

        # Extract index
        idx = indices[0] if len(indices) else None

        # If there is negative in s0-column -
        # look for negative in non-basis variables
        # else look for first positive in F-row
        jdx = self.find_negative(idx) if idx else self.find_positive()

        if jdx is None:
            return

        # No column - either solved, or not solvable at all
        idx = self.find_minimal_ratio(jdx) if jdx else None

        if idx is None:
            return

        # Return tuple if both of indices are valid
        return (idx, jdx)
    
    def step(self, idx: int, jdx: int):
        # Table dimensions
        x, y = self.table.shape

        # Adjust labels
        self.v_labels[idx], self.h_labels[jdx] = self.h_labels[jdx], self.v_labels[idx]

        # Solver-values shortcut
        solver = self.table[idx, jdx]

        # Flatten makes copies
        solver_row = self.table[idx, :].flatten()
        solver_column = self.table[:, jdx].flatten()

        # Is there any other way?
        for i, j in product(range(x), range(y)):
            if i == idx and j == jdx:
                self.table[i, j] = 1 / solver
            elif i == idx:
                self.table[i, j] /= solver
            elif j == jdx:
                self.table[i, j] /= -solver
            else:
                self.table[i, j] -= solver_column[i] * solver_row[j] / solver

    def table_to_md(self, precision: int = 3):
        return pd.DataFrame(
            np.round(self.table, precision),
            index=[f"${value}$" for value in self.v_labels],
            columns=[f"${value}$" for value in self.h_labels]
        ).to_markdown()
    
    def function_to_md(self, precision: int = 3):
        *basis, F = [
            f"{label}={value}" for value, label in zip(
                np.round(self.table[:, 0].flatten(), precision),
                self.v_labels
            )
        ]
        basis = ', '.join(basis)
        non_basis = '='.join(self.h_labels[1:])

        return '\n'.join(
            [
                f"$${F}$$",
                f"$${non_basis}=0$$",
                f"$${basis}$$"
            ]
        )


class Simplex:
    @classmethod
    def resolve(cls, c: np.ndarray, A: np.ndarray, b: np.ndarray, verbose: bool = False) -> Table:
        table = Table.create(c, A, b)

        if verbose:
            display_markdown("### Изначальная симплекс-таблица", raw=True)
            display_markdown(table.table_to_md(), raw=True)
            display_markdown(table.function_to_md(), raw=True)

        counter = count(1)
        while solver := table.find_solver():
            table.step(*solver)
            
            if verbose:
                display_markdown(f"### {next(counter)} шаг. Разрешающий элемент $x_{{{solver[0]},{solver[1]}}} = {table.table[solver]}$", raw=True)
                display_markdown(table.table_to_md(), raw=True)
                display_markdown(table.function_to_md(), raw=True)

        return table
