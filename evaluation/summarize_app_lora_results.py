import argparse
from pathlib import Path
from datetime import datetime


APPS = ["amazon", "ebay", "flipkart", "gmail", "clock", "reminder", "youtube"]
ROUND_LIST = [30]


def extract_metrics_from_file(path: Path) -> str:
    """从单个 *_result.txt 文件中抽取关键信息，返回用于写入 summary 的短文本。

    这里只保留整体的 Step-level accuracy 和 Episode-level accuracy。
    """
    if not path.is_file():
        return ""

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    selected = []
    for line in lines:
        if "Step-level accuracy" in line or "Episode-level accuracy" in line:
            selected.append(line)

    # 最多保留前两行（Step-level 和 Episode-level）
    return "\n".join(selected[:2])


def summarize(base_output_dir: Path, summary_file: Path):
    now = datetime.now().strftime("%a %b %d %H:%M:%S %Z %Y")

    with summary_file.open("w", encoding="utf-8") as f:
        f.write("App LoRA Evaluation Summary\n")
        f.write("================================\n")
        f.write(f"Base Output Directory: {base_output_dir}\n")
        f.write(f"Date: {now}\n\n")

        for model_app in APPS:
            model_root = base_output_dir / f"app_lora_{model_app}"
            f.write(f"Model LoRA App: {model_app}\n")
            f.write("--------------------------------\n")

            if not model_root.exists():
                f.write(f"  [WARNING] Model directory not found: {model_root}\n\n")
                continue

            for data_app in APPS:
                f.write(f"  Test Set App: {data_app}\n")

                any_round_found = False

                for round_id in ROUND_LIST:
                    # 在 model_root 下递归查找 global_lora_<round>/infer_result/<data_app> 下的 *_result.txt
                    pattern = f"**/global_lora_{round_id}/infer_result/{data_app}/*_result.txt"
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
                            f.write(
                                "        [WARNING] No accuracy lines found in this result file.\n"
                            )

                if not any_round_found:
                    f.write("    [INFO] No result files found for this test app.\n")

                f.write("\n")

            f.write("\n")

    print(f"[INFO] Summary report saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Summarize app LoRA evaluation results.")
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default="./lora_app",
        help="Base output directory that contains app_lora_* subdirectories.",
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default="./lora_app/app_lora_summary.txt",
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


