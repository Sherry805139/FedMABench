"""
按category分组数据，为每个category生成独立的训练数据集
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def load_category_mapping(category_file):
    """加载episode_id到category的映射"""
    with open(category_file, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    return {ep_id: data.get('category', 'Unknown') for ep_id, data in mapping.items()}


def split_data_by_category(input_jsonl, category_mapping_file, output_dir):
    """按category分组数据"""
    # 加载category映射
    episode_to_category = load_category_mapping(category_mapping_file)
    
    # 按category分组episodes
    category_episodes = defaultdict(list)
    
    print(f"Reading data from {input_jsonl}...")
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            try:
                episode = json.loads(line)
                episode_id = episode.get('episode_id', '')
                
                # 获取category
                category = episode_to_category.get(episode_id, 'Unknown')
                category_episodes[category].append(episode)
            except Exception as e:
                print(f"Error processing line: {e}")
                continue
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 为每个category保存数据
    print("\nSaving category-specific datasets...")
    for category, episodes in category_episodes.items():
        output_file = output_dir / f"{category.lower()}_train.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for episode in episodes:
                f.write(json.dumps(episode, ensure_ascii=False) + '\n')
        print(f"  {category}: {len(episodes)} episodes -> {output_file}")
    
    # 保存统计信息
    stats_file = output_dir / "category_stats.json"
    stats = {
        cat: len(episodes) for cat, episodes in category_episodes.items()
    }
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Statistics saved to: {stats_file}")
    return category_episodes


def main():
    parser = argparse.ArgumentParser(description="Split dataset by category")
    parser.add_argument(
        '--input_jsonl',
        type=str,
        required=True,
        help='Path to episode-wise-conversations.jsonl'
    )
    parser.add_argument(
        '--category_mapping',
        type=str,
        required=True,
        help='Path to episode_category_mapping.json'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data_by_category',
        help='Output directory for category-specific datasets'
    )
    
    args = parser.parse_args()
    
    split_data_by_category(
        args.input_jsonl,
        args.category_mapping,
        args.output_dir
    )


if __name__ == '__main__':
    main()



