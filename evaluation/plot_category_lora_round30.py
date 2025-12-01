import argparse
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


CATEGORIES = ["Shopping", "Traveling", "Office", "Lives", "Entertainment"]


def parse_round30_from_summary(summary_path: Path) -> Dict[str, Dict[str, Optional[float]]]:
    """从 category_lora_summary.txt 中解析每个 (model_cat, test_cat) 在 Round 30 的 step-level accuracy."""
    results: Dict[str, Dict[str, Optional[float]]] = {
        m: {t: None for t in CATEGORIES} for m in CATEGORIES
    }

    current_model: Optional[str] = None
    current_test_cat: Optional[str] = None
    current_round: Optional[int] = None

    with summary_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # 解析当前 model
            if line.startswith("Model LoRA Category:"):
                current_model = line.split(":", 1)[1].strip()
                current_test_cat = None
                current_round = None
                continue

            # 解析当前 test set category
            if line.startswith("Test Set Category:"):
                current_test_cat = line.split(":", 1)[1].strip()
                current_round = None
                continue

            # 解析 round
            if line.startswith("Round ") and line.endswith(":"):
                try:
                    round_str = line.split("Round", 1)[1].split(":", 1)[0].strip()
                    current_round = int(round_str)
                except Exception:
                    current_round = None
                continue

            # 解析 step-level accuracy（只保留 Round 30）
            if "Step-level accuracy:" in line and current_model and current_test_cat:
                if current_round == 30:
                    try:
                        acc_str = line.split("Step-level accuracy:", 1)[1].strip()
                        # 可能形如 "85.71%"，去掉百分号
                        if acc_str.endswith("%"):
                            acc_str = acc_str[:-1]
                        acc = float(acc_str)
                        if current_model in results and current_test_cat in results[current_model]:
                            results[current_model][current_test_cat] = acc
                    except Exception:
                        # 忽略解析失败
                        pass

    return results


def plot_heatmap(results: Dict[str, Dict[str, Optional[float]]], output_path: Path):
    """根据解析出的结果绘制 5x5 heatmap."""
    data = np.full((len(CATEGORIES), len(CATEGORIES)), np.nan, dtype=float)

    for i, model_cat in enumerate(CATEGORIES):
        for j, test_cat in enumerate(CATEGORIES):
            val = results.get(model_cat, {}).get(test_cat)
            if val is not None:
                data[i, j] = val

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap="viridis", vmin=0, vmax=100)

    # 坐标轴标签
    ax.set_xticks(np.arange(len(CATEGORIES)))
    ax.set_yticks(np.arange(len(CATEGORIES)))
    ax.set_xticklabels(CATEGORIES, rotation=45, ha="right")
    ax.set_yticklabels(CATEGORIES)
    ax.set_xlabel("Test Set Category")
    ax.set_ylabel("Model LoRA Category")
    ax.set_title("Round 30 Step-level Accuracy (%)")

    # 在每个格子里标数字
    for i in range(len(CATEGORIES)):
        for j in range(len(CATEGORIES)):
            if not np.isnan(data[i, j]):
                ax.text(
                    j,
                    i,
                    f"{data[i, j]:.1f}",
                    ha="center",
                    va="center",
                    color="white" if data[i, j] < 60 else "black",
                    fontsize=8,
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy (%)")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved heatmap to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Round 30 5x5 category LoRA heatmap from summary file.")
    parser.add_argument(
        "--summary_file",
        type=str,
        default="output/category_lora_summary.txt",
        help="Path to category_lora_summary.txt",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default="output/category_lora_round30_heatmap.png",
        help="Path to save heatmap image (PNG).",
    )

    args = parser.parse_args()
    summary_path = Path(args.summary_file)
    output_path = Path(args.output_png)

    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    results = parse_round30_from_summary(summary_path)
    plot_heatmap(results, output_path)


if __name__ == "__main__":
    main()


