"""
分析未映射的app，帮助用户快速添加到映射表
"""
import json
import argparse
from collections import defaultdict
from pathlib import Path


def load_mapping(mapping_file):
    """加载映射表"""
    with open(mapping_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return {
            k.lower(): v for k, v in data.items() 
            if not k.startswith('_') and isinstance(v, str)
        }


def analyze_unmapped_apps(episode_mapping_file, app_mapping_file):
    """分析未映射的app"""
    # 加载episode映射
    with open(episode_mapping_file, 'r', encoding='utf-8') as f:
        episode_data = json.load(f)
    
    # 加载app映射
    app_mapping = load_mapping(app_mapping_file)
    
    # 统计未映射的app
    unmapped = defaultdict(int)
    mapped = defaultdict(int)
    
    for ep_id, ep_info in episode_data.items():
        app_name = ep_info.get('app_name', '')
        category = ep_info.get('category', '')
        
        if category == 'Unknown':
            unmapped[app_name] += 1
        else:
            mapped[app_name] += 1
    
    print("=" * 80)
    print("Unmapped Apps Analysis")
    print("=" * 80)
    print(f"\nTotal episodes: {len(episode_data)}")
    print(f"Mapped episodes: {sum(mapped.values())}")
    print(f"Unmapped episodes: {sum(unmapped.values())}")
    print(f"Mapping rate: {sum(mapped.values()) / len(episode_data) * 100:.2f}%")
    
    print(f"\n📊 Unmapped apps ({len(unmapped)} unique apps):")
    print("-" * 80)
    
    # 按频率排序
    sorted_unmapped = sorted(unmapped.items(), key=lambda x: -x[1])
    
    # 生成建议的映射（基于相似性）
    print("\n💡 Suggested mappings (you may need to verify):")
    print("-" * 80)
    
    suggestions = []
    for app_name, count in sorted_unmapped[:100]:  # 只显示前100个
        app_lower = app_name.lower()
        # 尝试找到相似的已映射app
        similar_found = False
        for mapped_app in app_mapping.keys():
            if app_lower in mapped_app or mapped_app in app_lower:
                # 找到相似的，建议使用相同的category
                suggestions.append({
                    'app': app_name,
                    'count': count,
                    'suggested_category': app_mapping[mapped_app],
                    'similar_to': mapped_app
                })
                similar_found = True
                break
        
        if not similar_found:
            suggestions.append({
                'app': app_name,
                'count': count,
                'suggested_category': None,
                'similar_to': None
            })
    
    # 按category分组显示建议
    by_category = defaultdict(list)
    no_suggestion = []
    
    for sug in suggestions:
        if sug['suggested_category']:
            by_category[sug['suggested_category']].append(sug)
        else:
            no_suggestion.append(sug)
    
    for category in sorted(by_category.keys()):
        print(f"\n{category}:")
        for sug in by_category[category][:10]:  # 每个category最多显示10个
            print(f"  \"{sug['app']}\": \"{category}\",  # {sug['count']} episodes (similar to: {sug['similar_to']})")
    
    if no_suggestion:
        print(f"\n⚠️  Apps without suggestions ({len(no_suggestion)} apps):")
        for sug in no_suggestion[:20]:  # 只显示前20个
            print(f"  - {sug['app']} ({sug['count']} episodes)")
    
    # 保存建议到文件
    suggestions_file = Path(episode_mapping_file).parent / "suggested_app_mappings.json"
    suggestions_data = {
        'by_category': {cat: [s for s in sugs] for cat, sugs in by_category.items()},
        'no_suggestion': no_suggestion[:50]
    }
    with open(suggestions_file, 'w', encoding='utf-8') as f:
        json.dump(suggestions_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Suggestions saved to: {suggestions_file}")
    print("\n💡 Next steps:")
    print("   1. Review the suggestions above")
    print("   2. Add verified mappings to app_to_category_mapping.json")
    print("   3. Re-run extract_category_mapping.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze unmapped apps and suggest mappings")
    parser.add_argument(
        '--episode_mapping',
        type=str,
        default='./episode_category_mapping.json',
        help='Path to episode_category_mapping.json'
    )
    parser.add_argument(
        '--app_mapping',
        type=str,
        default='./data_process/app_to_category_mapping.json',
        help='Path to app_to_category_mapping.json'
    )
    
    args = parser.parse_args()
    analyze_unmapped_apps(args.episode_mapping, args.app_mapping)


