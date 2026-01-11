import pandas as pd
import numpy as np
import sympy as sp
import warnings
import time

warnings.filterwarnings('ignore', category=RuntimeWarning)

from expression import ExpressionGenerator

class Tokenizer:
    def __init__(self) -> None:
        self.token_map = {
            '<PAD>': 0, '<SOS>': 1, '<EOS>' : 2,
            'Add': 3, 'Mul': 4, 'Pow': 5,
            'sin': 6, 'cos': 7, 'exp': 8, 'log': 9, 'sqrt': 10, 'Abs': 11,
            'x': 12, 'CONST': 13
        }

    def expr_to_token_seq(self, expr):
        pass #TODO

    def token_seq_to_expr(self, tokens):
        pass #TODO

    

class DataGenerator:
    def __init__(self, max_depth: int, steps: int, const_prob: float, leaf_prob: float,
                 min_x: float, max_x: float, min_y: float, max_y: float) -> None:
        self.max_depth = max_depth
        self.steps = steps
        self.const_prob = const_prob
        self.generator = ExpressionGenerator(max_depth, const_prob, leaf_prob)
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def generate_points(self, expr):
        f = sp.lambdify(sp.Symbol('x'), expr, modules='numpy')
        x_values = np.linspace(num=self.steps, start=self.min_x, stop = self.max_x)
        try:
            y_values = f(x_values)
            if np.isscalar(y_values) or np.ndim(y_values) == 0:
                y_values = np.full_like(x_values, y_values)
            if np.iscomplexobj(y_values):
                return None
        except:
            return None
        mask = np.isfinite(y_values) & (y_values < self.max_y) & (y_values > self.min_y)
        if np.sum(mask) / np.size(mask) <= 0.3:
            return None
        dy = np.diff(y_values)
        diff_mask = mask[:-1] & mask[1:]
        if np.sum(diff_mask) < 2:
            return None
        valid_dy = dy[diff_mask]
        mean_change = np.mean(np.abs(valid_dy))
        if mean_change > 0.5:
            return None
        sign_changes = np.sum(np.diff(np.sign(valid_dy)) != 0)
        if sign_changes > len(valid_dy) * 0.3:
            return None
        y_values[~mask] = 0.0
        return np.vstack((x_values, y_values, mask.astype(float))).T

    def generate_data(self):
        pass #TODO


data_gen  = DataGenerator(max_depth=4, steps = 1000, const_prob=0.1, leaf_prob=0.2, min_x=-10, max_x=10, min_y=-10, max_y=10)
exprs = set()
attempts = 0
start = time.time()
while len(exprs) < 100 and attempts < 1000:
    attempts += 1
    expr = data_gen.generator.generate_expr()
    if expr is None:
        continue
    if expr.is_constant():
        continue
    if str(expr) in ['x', '-x'] and len(exprs) > 5:
        continue
    if data_gen.generate_points(expr) is None:
        continue
    expr_str = str(expr)
    if expr_str not in exprs:
        exprs.add(expr_str)
        print(f"{len(exprs)}. {expr_str}")

print(f"\nВсего сгенерировано уникальных неконстантных функций: {len(exprs)}")
print(f"Всего затрачено {time.time() - start}")