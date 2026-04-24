import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_prep.canonicalize import run_canonicalization
from src.data_prep.collisions import run_collision_removal
from src.data_prep.compile import run_compilation
from src.data_prep.normalize import run_normalization


def main():
    parser = argparse.ArgumentParser(description="Сборка датасета Graph2Eq")
    parser.add_argument(
        "--raw_data", type=str, default="data", help="Папка с сырыми чанками"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_cleared",
        help="Папка для .pt тензоров",
    )
    args = parser.parse_args()

    # Папки для промежуточных этапов
    raw_dir = Path(args.raw_data)
    step1_dir = Path("data/tmp_1_canonical")
    step2_dir = Path("data/tmp_2_normalized")
    step3_dir = Path("data/tmp_3_no_collisions")
    final_dir = Path(args.output_dir)

    print("\n" + "=" * 40)
    print("ШАГ 1: Канонизация выражений (cos -> sin)")
    print("=" * 40)
    run_canonicalization(raw_dir, step1_dir)

    print("\n" + "=" * 40)
    print("ШАГ 2: Геометрическая чистка и BBox нормализация")
    print("=" * 40)
    run_normalization(step1_dir, step2_dir)

    print("\n" + "=" * 40)
    print("ШАГ 3: Удаление визуальных коллизий (GPU)")
    print("=" * 40)
    ckpt_path = Path("data/gpu_checkpoint.pt")
    run_collision_removal(step2_dir, step3_dir, ckpt_path)

    print("\n" + "=" * 40)
    print("ШАГ 4: Компиляция PyTorch тензоров")
    print("=" * 40)
    run_compilation(step3_dir, final_dir)

    print("\n Датасет готов!")


if __name__ == "__main__":
    main()
