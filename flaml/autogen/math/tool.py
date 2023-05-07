from typing import Union, Any
from math import isclose
import func_timeout
from sympy.solvers import solve
from sympy import Symbol, Eq
import math
from sympy import simplify
import numpy as np
import cvxpy as cp
import statistics


def get_precision(gt_ans: float) -> int:
    precision = 5
    if "." in str(gt_ans):
        precision = len(str(gt_ans).split(".")[-1])
    return precision


def finqa_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = False,
    is_close: float = False,
) -> bool:
    if prediction is None:
        return False
    elif type(prediction) == bool:
        # bool questions
        if prediction:
            return reference == "yes"
        else:
            return reference == "no"
    elif type(reference) == str or type(prediction) == str:
        # string questions
        return prediction == reference
    else:
        # number questions
        if include_percentage:
            gt_result = [reference / 100, reference, reference * 100]
        else:
            gt_result = [reference]
        for item in gt_result:
            try:
                if is_close:
                    if isclose(item, prediction, rel_tol=0.001):
                        return True
                precision = min(get_precision(prediction), get_precision(item))
                if round(prediction, precision) == round(item, precision):
                    return True
            except Exception:
                continue
        return False


def simplify_ans(ans, convert_to_str: bool = True):
    if "relational" in str(type(ans)):
        return str(ans)
    elif "numpy" in str(type(ans)):
        if ans.shape == ():
            # scalar value
            ans = round(float(ans), 2)
        else:
            # array value
            ans = round(float(ans[0]), 2)
        if convert_to_str:
            return str(ans)
        else:
            return ans
    elif not ans:
        return None
    else:
        if type(ans) in [list, tuple]:
            if "sympy" in str(type(ans[0])):
                try:
                    ans = [round(float(x), 2) for x in ans]
                except Exception:
                    ans = [str(x) for x in ans]
            if len(ans) == 1:
                ans = ans[0]
        else:
            if "sympy" in str(type(ans)):
                try:
                    ans = round(float(ans), 2)
                except Exception:
                    ans = str(ans)
        if convert_to_str:
            return str(ans)
        else:
            return ans


def floatify_ans(ans):
    if ans is None:
        return None
    elif type(ans) == dict:
        ans = list(ans.values())[0]
    elif type(ans) == bool:
        ans = ans
    elif type(ans) in [list, tuple]:
        if not ans:
            return None
        else:
            try:
                ans = float(ans[0])
            except Exception:
                ans = str(ans[0])
    else:
        try:
            ans = float(ans)
        except Exception:
            ans = str(ans)
    return ans


def parse_api_result(result):
    to_return = []
    for idx, g in enumerate(result["choices"]):
        text = g["text"]
        logprob = sum(g["logprobs"]["token_logprobs"])
        to_return.append((text, logprob))
    to_return = sorted(to_return, key=lambda tup: tup[1], reverse=True)
    to_return = [r[0] for r in to_return]
    return to_return


def solve_it(equation, variable):
    solution = solve(equation, variable, dict=True)
    if not solution:
        if isinstance(variable, list):
            solution = {v: None for v in variable}
        else:
            solution = {variable: None}
        return solution
    else:
        solution = solution[0]
        return solution


def safe_execute(code_string: str, keys=None):
    def execute(x):
        try:
            exec(x)
            locals_ = locals()
            if keys is None:
                return locals_.get("ans", None)
            else:
                return [locals_.get(k, None) for k in keys]
        except Exception:
            return None

    try:
        ans = func_timeout.func_timeout(5, execute, args=(code_string,))
    except func_timeout.FunctionTimedOut:
        ans = None

    return ans


def synthesize_program(result: str, prefix: str) -> str:
    # program = prefix
    program = """
import math
import numpy as np
import sympy as sp # added

def solver():
"""

    for i, line in enumerate(result.split("\n")):
        if line == "":
            continue
        if i == 0:
            program += line + "\n"
        else:
            if line.startswith("    "):
                program += line + "\n"
            else:
                break
    program += "print(solver())"
    # program += 'ans = solver()'
    return program


# def synthesize_program(result: str, prefix: str) -> str:
#     program = prefix
#     for i, line in enumerate(result.split('\n')):
#         if line == '':
#             continue
#         if '\t' or '    ' not in line:
#             program += '    ' + line + '\n'
#         else:
#             program += line + '\n'

#     program += 'ans = solver()'
#     return program
