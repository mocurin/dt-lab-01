import numpy as np


class Brackets:
    plain = ''
    round = 'p'
    square = 'b'
    curly = 'B'
    pipes = 'v'
    double = 'V'


def array_to_latex(array: np.array, name: str = str(), brackets: Brackets = Brackets.plain):
    assert len(array.shape) == 1, f"Can not display array of shape: {array.shape}"

    row = ' & '.join(array.astype(str))
    line = name and f"{name} = " + f"\\begin{{{brackets}matrix}}\n{row}\n\\end{{{brackets}matrix}}"

    return f"${line}$"

def matrix_to_latex(array: np.array, name: str = str(), brackets: Brackets = Brackets.plain):
    assert len(array.shape) == 2, f"Can not display array of shape: {array.shape}"

    rows = '\\\\\n'.join(' & '.join(row.astype(str)) for row in array)
    line = name and f"{name} = " + f"\\begin{{{brackets}matrix}}\n{rows}\n\\end{{{brackets}matrix}}"

    return f"${line}$"
