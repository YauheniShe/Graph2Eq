import gzip
import logging
import multiprocessing as mp
import os
import pickle
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sympy as sp
from sympy.core.cache import clear_cache
from tqdm import tqdm

from src.expression import ExpressionGenerator
from src.tokenizer import InvalidExpressionError, Tokenizer, TokenizerError

BASE_DIR = Path(__file__).resolve().parent.parent
log_path = BASE_DIR / "new_data" / "data.log"
logging.basicConfig(filename=log_path, level=logging.ERROR)

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class GenConfig:
    max_ops: int
    timeout: int
    steps: int
    min_x: float
    max_x: float
    min_y: float
    max_y: float


def generate_points(expr, config: GenConfig):
    if expr.has(sp.zoo) or expr.has(sp.oo) or expr.has(sp.nan):
        return None
    f = sp.lambdify(sp.Symbol("x"), expr, modules="numpy")
    x_values = np.linspace(num=config.steps, start=config.min_x, stop=config.max_x)
    try:
        y_values = f(x_values)
        if np.isscalar(y_values) or np.ndim(y_values) == 0:
            y_values = np.full_like(x_values, y_values)
        if np.iscomplexobj(y_values):
            return None
    except Exception:
        return None
    mask = np.isfinite(y_values) & (y_values < config.max_y) & (y_values > config.min_y)
    if np.sum(mask) / np.size(mask) <= 0.3:
        return None
    dy = np.diff(y_values)
    diff_mask = mask[:-1] & mask[1:]
    if np.sum(diff_mask) < 2:
        return None
    valid_dy = dy[diff_mask]
    mean_change = np.mean(np.abs(valid_dy))
    if mean_change > 2:
        return None
    sign_changes = np.sum(np.diff(np.sign(valid_dy)) != 0)
    if sign_changes > len(valid_dy) * 0.3:
        return None
    return np.vstack((x_values[mask], y_values[mask])).T


def worker_task(args):
    config, seed = args

    random.seed(seed)
    np.random.seed(seed % (2**32))

    generator = ExpressionGenerator(config.max_ops, config.timeout)
    tokenizer = Tokenizer()

    attempts = 0
    while True:
        attempts += 1
        if attempts % 100000 == 0:
            clear_cache()
        try:
            skeleton, orig_expr, expr_instantiated = generator.generate_expr()
            try:
                token_seq = tokenizer.expr_to_token_seq(skeleton)
            except InvalidExpressionError:
                continue
            except TokenizerError:
                logging.exception(f"Ошибка токенизации: {skeleton}")
                continue

            points = generate_points(expr_instantiated, config)
            if points is None:
                continue

            return str(skeleton), str(expr_instantiated), token_seq, points

        except Exception:
            continue


class DataGenerator:
    def __init__(
        self,
        max_ops: int,
        steps: int,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        timeout: int,
        output_dir: str | Path,
    ) -> None:
        self.config = GenConfig(
            max_ops=max_ops,
            timeout=timeout,
            steps=steps,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = Tokenizer()

    def save_chunk(self, data, chunk_id):
        filename = self.output_dir / f"chunk_{chunk_id}.pkl.gz"
        with gzip.open(filename, "wb") as f:
            pickle.dump(data, f)

    def generate_data(
        self, size: int, chunk_size: int = 5000, n_jobs: int | None = None
    ):
        if n_jobs is None:
            n_jobs = mp.cpu_count() - 1

        existing_chunks = list(self.output_dir.glob("chunk_*.pkl.gz"))
        chunk_counter = 0
        if existing_chunks:
            max_id = -1
            for f in existing_chunks:
                try:
                    part = f.name.split("_")[1]
                    num = int(part.split(".")[0])
                    if num > max_id:
                        max_id = num
                except (IndexError, ValueError):
                    continue

            chunk_counter = max_id + 1
            print(
                f"Найдено {len(existing_chunks)} существующих файлов. Следующий чанк будет: {chunk_counter}"
            )

        print(f"Запуск генерации на {n_jobs} процессах...")
        start = time.time()

        buffer = []
        total_generated = 0

        def task_generator():
            seed_base = int.from_bytes(os.urandom(4), "little")
            for i in range(size * 10):
                yield (self.config, (seed_base + i) % (2**32))

        with mp.Pool(processes=n_jobs, maxtasksperchild=5000) as pool:
            result_iter = pool.imap_unordered(worker_task, task_generator())

            with tqdm(total=size, desc="Генерация", unit="expr") as pbar:
                for result in result_iter:
                    skeleton_str, instantiated_str, token_seq, points = result
                    item = {
                        "expr_str": skeleton_str,
                        "expr_instantiated_str": instantiated_str,
                        "tokens": token_seq,
                        "points": points,
                    }
                    buffer.append(item)
                    total_generated += 1

                    pbar.update(1)

                    if len(buffer) >= chunk_size:
                        self.save_chunk(buffer, chunk_counter)
                        chunk_counter += 1
                        buffer = []
                        tqdm.write(f"Сохранен чанк {chunk_counter - 1}")

                    if total_generated >= size:
                        pool.terminate()
                        break

                if buffer:
                    self.save_chunk(buffer, chunk_counter)

        print(f"\nВсего сгенерировано: {total_generated}")
        print(f"Всего затрачено {time.time() - start:.2f} сек")


def get_raw_polish_notation(expr):
    tokens = []
    for node in sp.preorder_traversal(expr):
        if node.args:
            tokens.append(node.func.__name__)
        else:
            tokens.append(str(node))

    return "[" + " ".join(tokens) + "]"


if __name__ == "__main__":
    print("\nГенерация трэйна...")
    train_gen = DataGenerator(
        max_ops=5,
        timeout=10,
        steps=500,
        min_x=-10,
        max_x=10,
        min_y=-10,
        max_y=10,
        output_dir=BASE_DIR / "new_data" / "train",
    )
    train_gen.generate_data(size=10000, chunk_size=100, n_jobs=8)

    val_gen = DataGenerator(
        max_ops=5,
        timeout=10,
        steps=500,
        min_x=-10,
        max_x=10,
        min_y=-10,
        max_y=10,
        output_dir=BASE_DIR / "new_data" / "val",
    )

    print("\nГенерация валидации...")
    val_gen.generate_data(size=9, chunk_size=100, n_jobs=8)
