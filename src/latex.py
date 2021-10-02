import numpy as np


class Brackets:
    plain = ''
    round = 'p'
    square = 'b'
    curly = 'B'
    pipes = 'v'
    double = 'V'


def array_to_latex(array: np.ndarray, name: str = str(), brackets: Brackets = Brackets.plain):
    assert len(array.shape) == 1, f"Can not display array of shape: {array.shape}"

    row = ' & '.join(array.astype(str))
    line = name and f"{name} = " + f"\\begin{{{brackets}matrix}}\n{row}\n\\end{{{brackets}matrix}}"

    return f"${line}$"

def matrix_to_latex(array: np.ndarray, name: str = str(), brackets: Brackets = Brackets.plain):
    assert len(array.shape) == 2, f"Can not display array of shape: {array.shape}"

    rows = '\\\\\n'.join(' & '.join(row.astype(str)) for row in array)
    line = name and f"{name} = " + f"\\begin{{{brackets}matrix}}\n{rows}\n\\end{{{brackets}matrix}}"

    return f"${line}$"

def equation_body(values: list, labels: list, precision: int = 3) -> str:
    return ''.join(
        f"{'+' if value > 0 else ''}{value if value != 1.0 else ''}{var}"
        for var, val in zip(labels, values)
        if (value := np.round(val, precision))
    ).lstrip('+')

def equation_system(values: np.ndarray, result: np.ndarray, labels: np.array, precision: int = 3) -> str:
    line = '\\\\\n'.join(f"{equation_body(val, labels, precision)}={res}" for val, res in zip(values, result))

    return f"$\\begin{{cases}}{line}\n\\end{{cases}}$"

def check(left: np.ndarray, right: np.ndarray, precision: int = 3) -> str:
    result = sum(left * right)

    # Округляем все значения
    left, right = [
        [
            np.round(value, precision)
            for value in values
        ]
        for values in [
            left, right
        ]
    ]

    # Соединяем при помощи cdot все пары значений без 0
    values = [
        '\cdot'.join(
            [
                f"({val})" if val < 0 else str(val)
                for val in values
            ]
        )
        for values in zip(
            left, right
        )
        if all(values)
    ]

    # Все суммируем, добавляем результат
    return '+'.join(values) + f"={np.round(result, precision)}"
