import argparse
from pathlib import Path
from datetime import datetime


CATEGORIES = ["Shopping", "Traveling", "Office", "Lives", "Entertainment"]
ROUND_LIST = [5, 10, 15, 20, 25, 30]


def extract_metrics_from_file(path: Path) -> str:
    """从单个 *_result.txt 文件中抽取关键信息，返回用于写入 summary 的短文本。

    按你的需求，这里只保留整体的 Step-level accuracy，
    不再展示各个 Category 的细粒度准确率。
    """
    if not path.is_file():
        return ""

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    selected = []
    for line in lines:
        # 只保留整体 step-level accuracy
        if "Step-level accuracy" in line:
            selected.append(line)

    return "\n".join(selected[:1])


def summarize(base_output_dir: Path, summary_file: Path):
    now = datetime.now().strftime("%a %b %d %H:%M:%S %Z %Y")

    with summary_file.open("w", encoding="utf-8") as f:
        f.write("Category LoRA Evaluation Summary\n")
        f.write("================================\n")
        f.write(f"Base Output Directory: {base_output_dir}\n")
        f.write(f"Date: {now}\n\n")

        for model_cat in CATEGORIES:
            model_lower = model_cat.lower()
            model_root = base_output_dir / f"category_lora_{model_lower}"
            f.write(f"Model LoRA Category: {model_cat}\n")
            f.write("--------------------------------\n")

            if not model_root.exists():
                f.write(f"  [WARNING] Model directory not found: {model_root}\n\n")
                continue

            for data_cat in CATEGORIES:
                data_lower = data_cat.lower()
                f.write(f"  Test Set Category: {data_cat}\n")

                any_round_found = False

                for round_id in ROUND_LIST:
                    # 兼容嵌套结构：在 model_root 下递归查找 global_lora_<round>/infer_result/<data_lower> 下的 *_result.txt
                    pattern = f"**/global_lora_{round_id}/infer_result/{data_lower}/*_result.txt"
                    result_files = sorted(model_root.glob(pattern))

                    if not result_files:
                        continue

                    any_round_found = True
                    f.write(f"    Round {round_id}:\n")
                    for result_file in result_files:
                        metrics = extract_metrics_from_file(result_file)
                        f.write(f"      {result_file.relative_to(base_output_dir)}:\n")
                        if metrics:
                            for line in metrics.splitlines():
                                f.write(f"        {line}\n")
                        else:
                            f.write("        [WARNING] No accuracy lines found in this result file.\n")

                if not any_round_found:
                    f.write("    [INFO] No result files found for this test category.\n")

                f.write("\n")

            f.write("\n")

    print(f"[INFO] Summary report saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Summarize category LoRA evaluation results.")
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default="./output",
        help="Base output directory that contains category_lora_* subdirectories.",
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default="./output/category_lora_summary.txt",
        help="Path to the summary txt file to generate.",
    )

    args = parser.parse_args()
    base_output_dir = Path(args.base_output_dir).resolve()
    summary_file = Path(args.summary_file).resolve()

    base_output_dir.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    summarize(base_output_dir, summary_file)


if __name__ == "__main__":
    main()


