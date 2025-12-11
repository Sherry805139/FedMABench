"""
从 episode-wise-conversations.jsonl 中提取 episode_id 到 app_name 的映射
用于按 app 进行数据集划分（每个 app 一个数据集）
"""
import json
import argparse
import re
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


# 只保留和训练相关的 8 个目标 app
TARGET_APPS = [
    "amazon",
    "ebay",
    "flipkart",
    "gmail",
    "clock",
    "google drive",
    "reminder",
    "youtube",
]


def extract_app_name_from_conversations(conversations):
    """从 conversations 中提取 app_name
    方法1: 从 assistant 的 value 中提取 "Open App: <app_name>"
    方法2: 如果方法1失败，从 user 的 value 中提取 "xxx app" 或 "xxx App"
    """
    if not isinstance(conversations, list):
        return None

    app_name = None

    # 方法1: 查找 assistant 的回复中的 "Open App"
    for msg in conversations:
        if isinstance(msg, dict) and msg.get("from") == "assistant":
            value = msg.get("value", "")
            if isinstance(value, str):
                # 匹配 "Open App: <app_name>" 模式
                match = re.search(r"Open App:\s*([^\n]+)", value, re.IGNORECASE)
                if match:
                    app_name = match.group(1).strip()
                    return app_name

    # 方法2: 如果 assistant 中没有，尝试从 user 的 instruction 中提取
    if not app_name:
        for msg in conversations:
            if isinstance(msg, dict) and msg.get("from") == "user":
                value = msg.get("value", "")
                if isinstance(value, str):
                    # 匹配 "xxx app" 或 "xxx App" 模式（不区分大小写）
                    patterns = [
                        r"(\w+(?:\s+\w+)*?)\s+app\b",  # "xxx app"
                        r"(\w+(?:\s+\w+)*?)\s+App\b",  # "xxx App"
                        r"app\s+(\w+(?:\s+\w+)*?)\b",  # "app xxx"
                        r"App\s+(\w+(?:\s+\w+)*?)\b",  # "App xxx"
                        r"the\s+(\w+(?:\s+\w+)*?)\s+app",  # "the xxx app"
                        r"in\s+the\s+(\w+(?:\s+\w+)*?)\s+app",  # "in the xxx app"
                        r"on\s+the\s+(\w+(?:\s+\w+)*?)\s+app",  # "on the xxx app"
                        r"using\s+the\s+(\w+(?:\s+\w+)*?)\s+app",  # "using the xxx app"
                        r"by\s+using\s+the\s+(\w+(?:\s+\w+)*?)\s+app",  # "by using the xxx app"
                    ]

                    for pattern in patterns:
                        match = re.search(pattern, value, re.IGNORECASE)
                        if match:
                            app_name = match.group(1).strip()
                            # 过滤掉一些常见的误匹配
                            if app_name.lower() not in [
                                "the",
                                "a",
                                "an",
                                "this",
                                "that",
                                "file",
                                "manager",
                            ]:
                                return app_name

                    # 特殊处理：匹配 "xxx.com app" 或 "xxx app"（带点）
                    match = re.search(r"(\w+(?:\.\w+)*)\s+app\b", value, re.IGNORECASE)
                    if match:
                        app_name = match.group(1).strip()
                        return app_name

    return app_name


def normalize_app_name(app_name):
    """规范化 app 名称，用于后续按 app 分组和文件命名"""
    if not app_name:
        return ""
    # 转换为小写，去除多余空格
    normalized = app_name.lower().strip()
    # 移除常见的后缀
    normalized = re.sub(r"\s+app$", "", normalized)
    normalized = re.sub(r"\s+application$", "", normalized)
    # 去掉开头的 in / in the / on the / at the 等前缀
    normalized = re.sub(
        r"^(in|on|at)\s+the\s+", "", normalized
    )  # in the xxx / on the xxx
    normalized = re.sub(r"^(in|on|at)\s+", "", normalized)  # in xxx / on xxx
    return normalized


def map_to_target_app(app_name):
    """将提取到的 app_name 映射到 8 个目标 app 之一
    如果无法映射，则返回 None（该 episode 将被丢弃）
    """
    if not app_name:
        return None
    norm = normalize_app_name(app_name)
    if not norm:
        return None

    # 直接相等
    for target in TARGET_APPS:
        if norm == target:
            return target

    # 模糊匹配：norm 包含 target，或者 target 包含 norm
    for target in TARGET_APPS:
        if target in norm:
            return target

    return None


def extract_app_names_from_data(jsonl_path):
    """从数据中提取所有 episode_id 到 app_name 的映射（只保留 8 个目标 app）
    - 只使用 conversations 信息进行提取：
      - 优先从 assistant 的 "Open App: <app_name>" 中获取
      - 否则从 user 文本里的 "xxx app" / "App xxx" 等 pattern 中获取
    - 然后将提取到的 app_name 映射到 [amazon, ebay, flipkart, gmail,
      clock, google drive, reminder, youtube] 这 8 个 app 之一
    - 只有能映射到这 8 个 app 的 episode 才会被保留
    """
    app_names = set()
    episode_to_app = {}
    error_count = 0
    sample_episode = None

    print(f"Reading data from {jsonl_path}...")
    if not Path(jsonl_path).exists():
        print(f"ERROR: File not found: {jsonl_path}")
        return app_names, episode_to_app

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(tqdm(f, desc="Processing episodes")):
            if line_num == 0:
                # 保存第一行用于调试
                sample_episode = line[:500] if len(line) > 500 else line

            try:
                episode = json.loads(line)
                episode_id = episode.get("episode_id", "")
                app_name = None

                # 只使用 conversations 信息提取 app_name
                if "conversations" in episode:
                    app_name = extract_app_name_from_conversations(
                        episode.get("conversations", [])
                    )

                # 如果找到了 app_name，尝试映射到目标 app
                target_app = None
                if app_name:
                    target_app = map_to_target_app(app_name)

                # 只保留成功映射到 8 个目标 app 的 episode
                if target_app:
                    app_names.add(target_app)
                    episode_to_app[episode_id] = {
                        "app_name": app_name,  # 原始提取到的名称（方便调试）
                        "normalized_app_name": target_app,  # 归一化为 8 个目标 app 之一
                    }

            except json.JSONDecodeError as e:
                error_count += 1
                if error_count <= 3:
                    print(f"\nERROR: Line {line_num+1} JSON decode error: {e}")
                    print(f"  Line preview (first 200 chars): {line[:200]}")
            except Exception as e:
                error_count += 1
                if error_count <= 3:
                    print(f"\nERROR: Line {line_num+1} unexpected error: {e}")
                    print(f"  Error type: {type(e).__name__}")

    if sample_episode:
        print(f"\nSample first line (first 500 chars):")
        print(sample_episode)

    if error_count > 0:
        print(f"\nTotal errors encountered: {error_count}")

    return app_names, episode_to_app


def save_episode_app_mapping(episode_to_app, output_path):
    """保存 episode_id 到 app_name 的映射"""
    # 统计 app 分布
    app_counts = defaultdict(int)
    normalized_app_counts = defaultdict(int)
    for data in episode_to_app.values():
        app = data.get("app_name", "Unknown")
        norm_app = data.get("normalized_app_name", "unknown")
        app_counts[app] += 1
        normalized_app_counts[norm_app] += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "episode_to_app": episode_to_app,
                "app_stats": {
                    "raw_app_counts": app_counts,
                    "normalized_app_counts": normalized_app_counts,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nEpisode-to-app mapping saved to: {output_path}")
    print(f"Total episodes with app: {len(episode_to_app)}")
    print(f"Total unique raw apps: {len(app_counts)}")
    print(f"Total unique normalized apps: {len(normalized_app_counts)}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract episode_id to app_name mapping from episode data"
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default="/home/hmpiao/hmpiao/xuerong/FedMABench/android_control_unpack/episode-wise-conversations.jsonl",
        help="Path to episode-wise-conversations.jsonl",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="./episode_app_mapping.json",
        help="Output path for episode-to-app mapping JSON file",
    )

    args = parser.parse_args()

    # 从数据中提取 app 信息
    app_names, episode_to_app = extract_app_names_from_data(args.input_jsonl)

    print(f"\nFound {len(app_names)} unique raw app names:")
    for app in sorted(app_names):
        print(f"  - {app}")

    # 保存 episode -> app 映射
    save_episode_app_mapping(episode_to_app, args.output_json)

    print(f"\n✅ Done! Episode-to-app mapping saved to: {args.output_json}")


if __name__ == "__main__":
    main()


