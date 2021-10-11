from dataclasses import dataclass
from itertools import product, count
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Table:
    table: np.ndarray
    h_labels: list
    v_labels: list

    @classmethod
    def create(cls, c: np.ndarray, A: np.ndarray, b: np.ndarray, sym: str = 'x', fsym='F'):
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

        # Create rows & column names
        _, columns = A.shape
        labels = [f"{sym}_{idx}" for idx in range(1, sum(A.shape) + 1)]
        labels = ['s_0', *labels[:columns]], [*labels[columns:], fsym]

        # Create helper structure
        return cls(table, *labels)
    
    def find_negative(self, idx: int) -> Optional[int]:
        # Values of non-basis variables in row of negative s0
        row = self.table[idx, 1:].flatten()

        # Look for first negative
        indices = np.flatnonzero(row < 0)

        # Extract index (compensate for missing leading value by adding one to index)
        return (indices[0] + 1) if len(indices) else None

    def find_positive(self) -> Optional[int]:
        # Values of non-basis variables in F-row (w/o s0-column value)
        row = self.table[-1, 1:].flatten()

        # Look for first positive
        indices = np.flatnonzero(row > 0)

        # Extract index (compensate for missing leading value by adding one to index)
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

        # Check if there were any valid ratios & return idx
        return idx if pos_inf[idx] != +np.inf else None

    def find_fixer(self) -> tuple[tuple[int, int], bool]:
        # Values of s0-column (w/o F-row value)
        column = self.table[:-1, 0].flatten()

        # Look for first negative
        indices = np.flatnonzero(column < 0)

        # Extract index
        idx = indices[0] if len(indices) else None

        if idx is None:
            return None, True

        jdx = self.find_negative(idx)

        # No column - either solved, or not solvable at all
        if jdx is None:
            return None, False

        # Find minimal positive ratio of solver s0 & solver column
        idx = self.find_minimal_ratio(jdx) if jdx else None

        # There is no solver row
        if idx is None:
            return None, False

        # Return tuple if both of indices are valid
        return (idx, jdx), True

    def find_solver(self) -> tuple[tuple[int, int], bool]:
        jdx = self.find_positive()

        # No column - solved
        if jdx is None:
            return None, True

        # Find minimal positive ratio of solver s0 & solver column
        idx = self.find_minimal_ratio(jdx) if jdx else None

        # There is no solver row
        if idx is None:
            return None, False

        # Return tuple if both of indices are valid
        return (idx, jdx), True
    
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
        
        return self

    def table_to_md(self, precision: int = 3):
        # Cast to pandas table with convinient methods
        return pd.DataFrame(
            np.round(self.table, precision),
            index=[f"${value}$" for value in self.v_labels],
            columns=[f"${value}$" for value in self.h_labels]
        ).to_markdown()
    
    def function_to_md(self, inverse: bool = False, precision: int = 3):
        # Group by basis & non-basis variables

        if inverse:
            self.table[-1, 0] *= -1

        *basis, F = [
            f"{label}={value}" for value, label in zip(
                np.round(self.table[:, 0].flatten(), precision),
                self.v_labels
            )
        ]
        non_basis = '='.join(self.h_labels[1:])
        variables = ', '.join([f"{non_basis}=0", *basis])

        if inverse:
            self.table[-1, 0] *= -1

        return [
                f"${F}$",
                f"${variables}$",
        ]


class Simplex:
    @classmethod
    def fix(cls, table: Table) -> bool:
        """First step of simplex-algorithm"""
        # Exhaust generator till final table
        for table, _, flag in cls.fix_gen(table):
            pass

        return flag
    
    @classmethod
    def fix_gen(cls, table: Table) -> tuple[Table, tuple[int, int], bool]:
        # Continiously check for condition №1
        while fixer := table.find_fixer():
            fixer, fixable = fixer

            # This won't be solvable if there is no
            # negative value in corresponding row
            # So, exit early
            if not fixer or not fixable:
                return
            # If there is negative value in corresponding column
            table.step(*fixer)

            # Table-after-fix
            yield table, fixer, fixable
    
    @classmethod
    def solve(cls, table: Table) -> bool:
        """First step of simplex-algorithm"""
        # Exhaust generator till final table
        flag = False
        for table, _, flag in cls.fix_gen(table):
            pass

        if not flag:
            return flag

        for table, _, flag in cls.solve_gen(table):
            pass

        return flag

    @classmethod
    def solve_gen(cls, table: Table) -> tuple[Table, tuple[int, int], bool]:
        # Continuosly check for condtion №2
        while solver := table.find_solver():
            solver, solvable = solver

            # Решение в тупике
            if not solvable or solver is None:
                return

            # Evolve table each time solver (col/row indicies pair) is found
            table.step(*solver)

            # Table-after-step
            yield table, solver, solvable
