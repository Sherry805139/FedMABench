"""
按 app 分组数据，为每个 app 生成独立的训练数据集
"""
import json
import argparse
import re
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def load_app_mapping(app_mapping_file):
    """加载 episode_id 到 app_name 的映射

    支持两种格式：
    1）新脚本生成的 episode_app_mapping.json：
       {
         "episode_to_app": {
           "<episode_id>": {
             "app_name": "...",
             "normalized_app_name": "..."
           },
           ...
         },
         "app_stats": { ... }
       }
    2）简单的 episode_id -> app_name 映射：
       {
         "<episode_id>": "app_name",
         ...
       }
    """
    with open(app_mapping_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 情况1：包含 "episode_to_app" 顶层键（推荐）
    if isinstance(data, dict) and "episode_to_app" in data:
        mapping = data["episode_to_app"]
        episode_to_app = {}
        for ep_id, v in mapping.items():
            if isinstance(v, dict):
                # 优先使用 normalized_app_name 作为分组 key
                app_name = v.get("normalized_app_name") or v.get("app_name", "Unknown")
            else:
                app_name = str(v)
            episode_to_app[ep_id] = app_name
        return episode_to_app

    # 情况2：直接是 episode_id -> app_name
    episode_to_app = {}
    for ep_id, v in data.items():
        if isinstance(v, dict):
            app_name = v.get("normalized_app_name") or v.get("app_name", "Unknown")
        else:
            app_name = str(v)
        episode_to_app[ep_id] = app_name
    return episode_to_app


def sanitize_app_name(app_name):
    """将 app 名称转换为适合文件名的形式"""
    if not app_name:
        return "unknown"
    name = app_name.lower().strip()
    # 去掉常见后缀
    name = re.sub(r"\s+app$", "", name)
    name = re.sub(r"\s+application$", "", name)
    # 替换空格和斜杠等为下划线
    name = re.sub(r"[\/\\\s]+", "_", name)
    # 移除其它不合法字符
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)
    if not name:
        name = "unknown"
    return name


def split_data_by_app(input_jsonl, app_mapping_file, output_dir):
    """按 app 分组数据"""
    # 加载 app 映射
    episode_to_app = load_app_mapping(app_mapping_file)

    # 按 app 分组 episodes
    app_episodes = defaultdict(list)

    print(f"Reading data from {input_jsonl}...")
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading episodes"):
            try:
                episode = json.loads(line)
                episode_id = episode.get("episode_id", "")

                # 获取 app 名称（已是 normalized 形式）
                app_name = episode_to_app.get(episode_id, "unknown")
                app_episodes[app_name].append(episode)
            except Exception as e:
                print(f"Error processing line: {e}")
                continue

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 为每个 app 保存数据
    print("\nSaving app-specific datasets...")
    for app_name, episodes in app_episodes.items():
        safe_name = sanitize_app_name(app_name)
        output_file = output_dir / f"{safe_name}_train.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for episode in episodes:
                f.write(json.dumps(episode, ensure_ascii=False) + "\n")
        print(f"  {app_name}: {len(episodes)} episodes -> {output_file}")

    # 保存统计信息
    stats_file = output_dir / "app_stats.json"
    stats = {app: len(episodes) for app, episodes in app_episodes.items()}
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Statistics saved to: {stats_file}")
    print(f"Total apps: {len(app_episodes)}")
    return app_episodes


def main():
    parser = argparse.ArgumentParser(description="Split dataset by app")
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Path to episode-wise-conversations.jsonl",
    )
    parser.add_argument(
        "--app_mapping",
        type=str,
        required=True,
        help="Path to episode_app_mapping.json (episode_id -> app mapping)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data_by_app",
        help="Output directory for app-specific datasets",
    )

    args = parser.parse_args()

    split_data_by_app(args.input_jsonl, args.app_mapping, args.output_dir)


if __name__ == "__main__":
    main()


