import numpy as np

from enum import Enum


class Brackets(str, Enum):
    plain = ""
    round = "p"
    square = "b"
    curly = "B"
    pipes = "v"
    double = "V"


def array_to_latex(
    array: np.ndarray,
    name: str = str(),
    precision: int = 2,
    brackets: Brackets = Brackets.plain,
):
    assert len(array.shape) == 1, f"Can not display array of shape: {array.shape}"

    if isinstance(brackets, str):
        brackets = Brackets(brackets)

    array = np.round(array, precision)
    row = " & ".join(
        str(value.astype(int)) if np.float64.is_integer(value) else value.astype(str)
        for value in array
    )
    line = (
        name
        and f"{name} = "
        + f"\\begin{{{brackets.value}matrix}}\n{row}\n\\end{{{brackets.value}matrix}}"
    )

    return f"${line}$"


def matrix_to_latex(
    array: np.ndarray,
    name: str = str(),
    precision: int = 2,
    brackets: Brackets = Brackets.plain,
):
    assert len(array.shape) == 2, f"Can not display array of shape: {array.shape}"

    if isinstance(brackets, str):
        brackets = Brackets(brackets)

    array = np.round(array, precision)
    rows = "\\\\\n".join(
        " & ".join(
            str(value.astype(int))
            if np.float64.is_integer(value)
            else value.astype(str)
            for value in row
        )
        for row in array
    )
    line = (
        name
        and f"{name} = "
        + f"\\begin{{{brackets.value}matrix}}\n{rows}\n\\end{{{brackets.value}matrix}}"
    )

    return f"${line}$"


def format_value(val: np.float64) -> str:
    if np.float64.is_integer(val):
        val = val.astype(int)

    absed = np.abs(val)

    return ("-" if val < 0 else "") + (absed.astype(str) if absed != 1 else "")


def equation_body(values: list, labels: list, precision: int = 3) -> str:
    return "+".join(
        f"{format_value(value)}{var}"
        for var, val in zip(labels, values)
        if (value := np.round(val, precision))
    ).replace("+-", "-")


def equation_system(
    values: np.ndarray, result: np.ndarray, labels: np.array, precision: int = 3
) -> str:
    line = "\\\\\n".join(
        f"{equation_body(val, labels, precision)} = {res.astype(int) if np.float64.is_integer(res) else res}"
        for val, res in zip(values, result)
    )

    return f"$\\begin{{cases}}{line}\n\\end{{cases}}$"


def check(left: np.ndarray, right: np.ndarray, precision: int = 3) -> str:
    result = sum(left * right)

    # ?????????????????? ?????? ????????????????
    left, right = [
        [np.round(value, precision) for value in values] for values in [left, right]
    ]

    # ?????????????????? ?????? ???????????? cdot ?????? ???????? ???????????????? ?????? 0
    values = [
        "\cdot".join([f"({val})" if val < 0 else str(val) for val in values])
        for values in zip(left, right)
        if all(values)
    ]

    # ?????? ??????????????????, ?????????????????? ??????????????????
    return "+".join(values) + f"={np.round(result, precision)}"
