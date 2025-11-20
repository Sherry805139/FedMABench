"""
从episode-wise-conversations.jsonl中提取app_name到category的映射
需要手动创建app_name到category的映射规则，或者从已有数据中推断
"""
import json
import argparse
import re
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


def extract_app_name_from_conversations(conversations):
    """从conversations中提取app_name
    方法1: 从assistant的value中提取 "Open App: <app_name>"
    方法2: 如果方法1失败，从user的value中提取 "xxx app" 或 "xxx App"
    """
    if not isinstance(conversations, list):
        return None
    
    app_name = None
    
    # 方法1: 查找assistant的回复中的"Open App"
    for msg in conversations:
        if isinstance(msg, dict) and msg.get('from') == 'assistant':
            value = msg.get('value', '')
            if isinstance(value, str):
                # 匹配 "Open App: <app_name>" 模式
                match = re.search(r'Open App:\s*([^\n]+)', value, re.IGNORECASE)
                if match:
                    app_name = match.group(1).strip()
                    return app_name
    
    # 方法2: 如果assistant中没有，尝试从user的instruction中提取
    if not app_name:
        for msg in conversations:
            if isinstance(msg, dict) and msg.get('from') == 'user':
                value = msg.get('value', '')
                if isinstance(value, str):
                    # 匹配 "xxx app" 或 "xxx App" 模式（不区分大小写）
                    # 匹配模式：单词 + "app"（可能有大写）
                    patterns = [
                        r'(\w+(?:\s+\w+)*?)\s+app\b',  # "xxx app"
                        r'(\w+(?:\s+\w+)*?)\s+App\b',  # "xxx App"
                        r'app\s+(\w+(?:\s+\w+)*?)\b',  # "app xxx"
                        r'App\s+(\w+(?:\s+\w+)*?)\b',  # "App xxx"
                        r'the\s+(\w+(?:\s+\w+)*?)\s+app',  # "the xxx app"
                        r'in\s+the\s+(\w+(?:\s+\w+)*?)\s+app',  # "in the xxx app"
                        r'on\s+the\s+(\w+(?:\s+\w+)*?)\s+app',  # "on the xxx app"
                        r'using\s+the\s+(\w+(?:\s+\w+)*?)\s+app',  # "using the xxx app"
                        r'by\s+using\s+the\s+(\w+(?:\s+\w+)*?)\s+app',  # "by using the xxx app"
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, value, re.IGNORECASE)
                        if match:
                            app_name = match.group(1).strip()
                            # 过滤掉一些常见的误匹配
                            if app_name.lower() not in ['the', 'a', 'an', 'this', 'that', 'file', 'manager']:
                                return app_name
                    
                    # 特殊处理：匹配 "xxx.com app" 或 "xxx app"（带点）
                    match = re.search(r'(\w+(?:\.\w+)*)\s+app\b', value, re.IGNORECASE)
                    if match:
                        app_name = match.group(1).strip()
                        return app_name
    
    return app_name


def extract_app_name_from_goal(goal):
    """从goal字段中使用正则表达式提取app名称（策略2）"""
    if not goal or not isinstance(goal, str):
        return None
    
    # 使用正则表达式匹配 "the xxx app" 模式
    pattern = re.compile(r'\bthe\s+(\w+(?:\s+\w+)?)\s+app\b', re.IGNORECASE)
    match = pattern.search(goal)
    
    if match:
        app_name = match.group(1).strip()
        return app_name
    
    return None


def extract_app_names_from_data(jsonl_path):
    """从数据中提取所有出现的app_name
    使用双策略方法：
    策略1: 如果actions中包含"open_app"动作，直接从app_name字段提取并清理
    策略2: 如果没有open_app动作，从goal字段使用正则表达式提取
    """
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
                app_name = None
                
                # 策略1: 检查actions中是否有"open_app"动作
                # 首先尝试从acts_origin中提取（旧格式）
                if 'acts_origin' in episode:
                    acts_origin = episode.get('acts_origin', [])
                    if isinstance(acts_origin, list):
                        for act_str in acts_origin:
                            try:
                                if isinstance(act_str, str):
                                    act = json.loads(act_str)
                                else:
                                    act = act_str
                                
                                if isinstance(act, dict) and act.get('action_type') == 'open_app':
                                    app_name = act.get('app_name', '')
                                    if app_name:
                                        # 清理字符串，去除BOM字符等
                                        app_name = app_name.replace('\ufeff', '').strip()
                                        break
                            except:
                                continue
                
                # 如果策略1失败，尝试策略2: 从goal字段提取
                if not app_name:
                    goal = episode.get('goal') or episode.get('instruction')
                    if goal:
                        app_name = extract_app_name_from_goal(goal)
                
                # 如果策略1和策略2都失败，尝试从conversations中提取（新格式，作为后备）
                if not app_name and 'conversations' in episode:
                    app_name = extract_app_name_from_conversations(episode.get('conversations', []))
                
                # 如果找到了app_name，记录它
                if app_name:
                    app_names.add(app_name)
                    if episode_id not in episode_to_app:
                        episode_to_app[episode_id] = app_name.lower()
                
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


def normalize_app_name(app_name):
    """规范化app名称，用于匹配"""
    if not app_name:
        return ""
    # 转换为小写，去除多余空格
    normalized = app_name.lower().strip()
    # 移除常见的后缀
    normalized = re.sub(r'\s+app$', '', normalized)
    normalized = re.sub(r'\s+application$', '', normalized)
    return normalized


def find_category_for_app(app_name, app_to_category_mapping):
    """查找app对应的category，支持多种匹配方式"""
    if not app_name:
        return None
    
    # 方法1: 直接匹配（原始大小写）
    if app_name in app_to_category_mapping:
        return app_to_category_mapping[app_name]
    
    # 方法2: 小写匹配
    app_lower = app_name.lower()
    if app_lower in app_to_category_mapping:
        return app_to_category_mapping[app_lower]
    
    # 方法3: 规范化匹配（去除"app"后缀等）
    normalized = normalize_app_name(app_name)
    if normalized in app_to_category_mapping:
        return app_to_category_mapping[normalized]
    
    # 方法4: 部分匹配（如果app_name包含映射表中的key）
    for mapped_app, category in app_to_category_mapping.items():
        if normalized in mapped_app.lower() or mapped_app.lower() in normalized:
            return category
    
    return None


def create_category_mapping(episode_to_app, app_to_category_mapping, output_path):
    """创建episode_id到category的映射文件"""
    episode_to_category = {}
    unmapped_apps = defaultdict(int)  # 统计未映射的app出现次数
    
    for episode_id, app_name in episode_to_app.items():
        category = find_category_for_app(app_name, app_to_category_mapping)
        
        if category:
            episode_to_category[episode_id] = {
                "app_name": app_name,
                "category": category
            }
        else:
            unmapped_apps[app_name] += 1
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
        print(f"\n⚠️  Warning: {len(unmapped_apps)} unique apps without category mapping:")
        print(f"Total unmapped episodes: {sum(unmapped_apps.values())}")
        print("\nTop 50 unmapped apps (by frequency):")
        for app, count in sorted(unmapped_apps.items(), key=lambda x: -x[1])[:50]:
            print(f"  - {app} ({count} episodes)")
        if len(unmapped_apps) > 50:
            print(f"  ... and {len(unmapped_apps) - 50} more apps")
        print("\n💡 Tip: You can add these apps to app_to_category_mapping.json")
        print("   Or check if they need normalization (e.g., 'File Manager' vs 'file manager app')")
    
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

