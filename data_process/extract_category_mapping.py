"""
从episode-wise-conversations.jsonl中提取app_name到category的映射
需要手动创建app_name到category的映射规则，或者从已有数据中推断
"""
import json
import argparse
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# App名称到Category的映射（基于论文Table 5）
# 完整的映射表保存在 app_to_category_mapping.json 文件中
# 这里只保留一些常见示例，实际使用时从JSON文件加载
APP_TO_CATEGORY_MAPPING = {
    # 示例映射（完整映射在app_to_category_mapping.json中）
    "amazon": "Shopping",
    "ebay": "Shopping",
    "flipkart": "Shopping",
    "kayak": "Traveling",
    "booking.com": "Traveling",
    "expedia": "Traveling",
    "gmail": "Office",
    "google docs": "Office",
    "google drive": "Office",
    "plantum": "Lives",
    "google fit": "Lives",
    "fitbit": "Lives",
    "youtube": "Entertainment",
    "spotify": "Entertainment",
    "netflix": "Entertainment",
}

# 如果映射文件不存在，需要手动创建或从数据中推断
CATEGORIES = ["Shopping", "Traveling", "Office", "Lives", "Entertainment"]


def extract_app_names_from_data(jsonl_path):
    """从数据中提取所有出现的app_name"""
    app_names = set()
    episode_to_app = {}
    error_count = 0
    sample_episode = None
    
    print(f"Reading data from {jsonl_path}...")
    if not Path(jsonl_path).exists():
        print(f"ERROR: File not found: {jsonl_path}")
        return app_names, episode_to_app
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Processing episodes")):
            if line_num == 0:
                # 保存第一行用于调试
                sample_episode = line[:500] if len(line) > 500 else line
            
            try:
                episode = json.loads(line)
                episode_id = episode.get('episode_id', '')
                
                # 检查字段是否存在
                if 'acts_origin' not in episode:
                    if line_num < 3:
                        print(f"\nWarning: Line {line_num+1} missing 'acts_origin' field")
                        print(f"Available fields: {list(episode.keys())}")
                    continue
                
                acts_origin = episode.get('acts_origin', [])
                
                if not isinstance(acts_origin, list):
                    if line_num < 3:
                        print(f"\nWarning: Line {line_num+1} 'acts_origin' is not a list: {type(acts_origin)}")
                    continue
                
                # 从acts_origin中提取app_name
                found_app = False
                for act_idx, act_str in enumerate(acts_origin):
                    try:
                        # 尝试解析JSON字符串
                        if isinstance(act_str, str):
                            act = json.loads(act_str)
                        else:
                            act = act_str
                        
                        if isinstance(act, dict) and act.get('action_type') == 'open_app':
                            app_name = act.get('app_name', '')
                            if app_name:
                                app_names.add(app_name)
                                # 记录每个episode对应的app（使用第一个open_app）
                                if episode_id not in episode_to_app:
                                    episode_to_app[episode_id] = app_name.lower()
                                    found_app = True
                                    break
                    except json.JSONDecodeError as e:
                        if line_num < 3 and act_idx < 2:
                            print(f"\nWarning: Line {line_num+1}, act {act_idx} JSON decode error: {e}")
                            print(f"  Act string (first 100 chars): {str(act_str)[:100]}")
                        continue
                    except Exception as e:
                        if line_num < 3:
                            print(f"\nWarning: Line {line_num+1}, act {act_idx} error: {e}")
                        continue
                
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


def create_category_mapping(episode_to_app, app_to_category_mapping, output_path):
    """创建episode_id到category的映射文件"""
    episode_to_category = {}
    unmapped_apps = set()
    
    for episode_id, app_name in episode_to_app.items():
        # 尝试直接匹配，也尝试小写匹配
        category = app_to_category_mapping.get(app_name) or app_to_category_mapping.get(app_name.lower())
        if category:
            episode_to_category[episode_id] = {
                "app_name": app_name,
                "category": category
            }
        else:
            unmapped_apps.add(app_name)
            # 如果没有映射，标记为Unknown
            episode_to_category[episode_id] = {
                "app_name": app_name,
                "category": "Unknown"
            }
    
    # 保存映射文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(episode_to_category, f, indent=2, ensure_ascii=False)
    
    print(f"\nCategory mapping saved to: {output_path}")
    print(f"Total episodes: {len(episode_to_category)}")
    
    # 统计category分布
    category_counts = defaultdict(int)
    for ep_data in episode_to_category.values():
        category_counts[ep_data['category']] += 1
    
    print("\nCategory distribution:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count} episodes")
    
    if unmapped_apps:
        print(f"\n⚠️  Warning: {len(unmapped_apps)} apps without category mapping:")
        for app in sorted(unmapped_apps):
            print(f"  - {app}")
        print("\nPlease update APP_TO_CATEGORY_MAPPING in this script.")
    
    return episode_to_category


def main():
    parser = argparse.ArgumentParser(description="Extract category mapping from episode data")
    parser.add_argument(
        '--input_jsonl',
        type=str,
        default='/home/hmpiao/hmpiao/xuerong/FedMABench/android_control_unpack/episode-wise-conversations.jsonl',
        help='Path to episode-wise-conversations.jsonl'
    )
    parser.add_argument(
        '--output_json',
        type=str,
        default='./episode_category_mapping.json',
        help='Output path for category mapping JSON file'
    )
    parser.add_argument(
        '--app_mapping_json',
        type=str,
        default='./app_to_category_mapping.json',
        help='Path to app_name to category mapping JSON file (default: ./app_to_category_mapping.json)'
    )
    
    args = parser.parse_args()
    
    # 优先使用指定的app映射文件，如果不存在则使用默认映射
    app_mapping_path = Path(__file__).parent / 'app_to_category_mapping.json'
    if args.app_mapping_json and Path(args.app_mapping_json).exists():
        app_mapping_path = Path(args.app_mapping_json)
    elif app_mapping_path.exists():
        print(f"Using default app mapping file: {app_mapping_path}")
    else:
        print(f"Warning: App mapping file not found: {app_mapping_path}")
        print("Using built-in mapping (limited apps)")
        app_mapping_path = None
    
    if app_mapping_path:
        print(f"Loading app mapping from {app_mapping_path}...")
        with open(app_mapping_path, 'r', encoding='utf-8') as f:
            app_mapping_data = json.load(f)
            # 过滤掉以_开头的注释键
            app_to_category_mapping = {
                k.lower(): v for k, v in app_mapping_data.items() 
                if not k.startswith('_') and isinstance(v, str)
            }
        print(f"Loaded {len(app_to_category_mapping)} app-to-category mappings")
    else:
        app_to_category_mapping = APP_TO_CATEGORY_MAPPING
    
    # 从数据中提取app信息
    app_names, episode_to_app = extract_app_names_from_data(args.input_jsonl)
    
    print(f"\nFound {len(app_names)} unique apps:")
    for app in sorted(app_names):
        print(f"  - {app}")
    
    # 创建category映射
    episode_to_category = create_category_mapping(
        episode_to_app,
        app_to_category_mapping,
        args.output_json
    )
    
    print(f"\n✅ Done! Category mapping saved to: {args.output_json}")


if __name__ == '__main__':
    main()

