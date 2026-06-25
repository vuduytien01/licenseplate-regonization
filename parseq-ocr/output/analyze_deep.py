import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def deep_analysis(output_dir='/kaggle/working/eval_no_gt'):
    """Phân tích sâu kết quả đánh giá."""
    
    reliability_csv = f'{output_dir}/reliability_assessment.csv'
    
    if not os.path.exists(reliability_csv):
        print(f"❌ Không tìm thấy file: {reliability_csv}")
        print("   Hãy chạy Cell 1 trước!")
        return None
    
    df = pd.read_csv(reliability_csv)
    
    print("="*80)
    print("🔍 PHÂN TÍCH SÂU KẾT QUẢ ĐÁNH GIÁ")
    print("="*80)
    
    # Chia thành 3 nhóm
    high_conf = df[df['reliability_score'] >= 80]
    medium_conf = df[(df['reliability_score'] >= 60) & (df['reliability_score'] < 80)]
    low_conf = df[df['reliability_score'] < 60]
    
    print("\n📊 PHÂN TÍCH ĐỘ TIN CẬY:")
    print("-"*80)
    print(f"  • High confidence (≥80): {len(high_conf):4d} samples ({len(high_conf)/len(df)*100:5.1f}%)")
    print(f"  • Medium confidence (60-79): {len(medium_conf):4d} samples ({len(medium_conf)/len(df)*100:5.1f}%)")
    print(f"  • Low confidence (<60): {len(low_conf):4d} samples ({len(low_conf)/len(df)*100:5.1f}%)")
    
    # Phân tích issues
    print("\n⚠️  PHÂN TÍCH CÁC VẤN ĐỀ:")
    print("-"*80)
    
    all_issues = []
    for issues in df['issues']:
        if issues != 'None':
            all_issues.extend([i.strip() for i in issues.split(',')])
    
    if all_issues:
        issue_counts = Counter(all_issues)
        
        print("  Top 10 vấn đề phổ biến nhất:")
        for i, (issue, count) in enumerate(issue_counts.most_common(10), 1):
            pct = count / len(df) * 100
            print(f"    {i:2d}. {issue:30s}: {count:4d} lần ({pct:5.1f}%)")
    
    # Phân tích predictions đáng ngờ
    print("\n🚨 PHÂN TÍCH PREDICTIONS ĐÁNG NGỜ (Score < 50):")
    print("-"*80)
    
    suspicious = df[df['reliability_score'] < 50]
    
    if len(suspicious) > 0:
        print(f"  Tổng số predictions đáng ngờ: {len(suspicious)}")
        
        print(f"\n  📏 Độ dài plate:")
        print(f"    • Trung bình: {suspicious['plate_length'].mean():.1f} ký tự")
        if len(suspicious) > 0:
            print(f"    • Phổ biến nhất: {suspicious['plate_length'].mode()[0]} ký tự")
        
        print(f"\n  📐 Aspect ratio:")
        print(f"    • Trung bình: {suspicious['aspect_ratio'].mean():.2f}")
        print(f"    • Range: {suspicious['aspect_ratio'].min():.2f} - {suspicious['aspect_ratio'].max():.2f}")
        
        print(f"\n  🔴 Top 10 predictions đáng ngờ nhất:")
        top_suspicious = suspicious.nsmallest(10, 'reliability_score')
        for idx, row in top_suspicious.iterrows():
            print(f"    • {row['image']:40s} | Score: {row['reliability_score']:3.0f} | "
                  f"Plate: '{row['plate']}' | Length: {row['plate_length']}")
    else:
        print("  ✅ Không có predictions đáng ngờ (score < 50)")
    
    # So sánh High vs Low
    print("\n📈 SO SÁNH HIGH vs LOW CONFIDENCE:")
    print("-"*80)
    
    if len(high_conf) > 0 and len(low_conf) > 0:
        metrics = ['plate_length', 'aspect_ratio', 'bbox_area']
        
        print(f"\n  {'Metric':<20s} | {'High Conf (≥80)':<20s} | {'Low Conf (<60)':<20s}")
        print("  " + "-"*70)
        
        for metric in metrics:
            high_mean = high_conf[metric].mean()
            low_mean = low_conf[metric].mean()
            print(f"  {metric:<20s} | {high_mean:>15.2f} | {low_mean:>15.2f}")
    
    # Trực quan hóa
    print("\n📊 Đang tạo biểu đồ phân tích sâu...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📊 Deep Analysis: High vs Low Confidence Predictions', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Plate length distribution
        if len(high_conf) > 0:
            axes[0, 0].hist(high_conf['plate_length'], bins=range(1, 15), 
                           alpha=0.6, label='High Confidence', color='#2ecc71', edgecolor='black')
        if len(low_conf) > 0:
            axes[0, 0].hist(low_conf['plate_length'], bins=range(1, 15), 
                           alpha=0.6, label='Low Confidence', color='#e74c3c', edgecolor='black')
        axes[0, 0].set_xlabel('Plate Length', fontsize=12)
        axes[0, 0].set_ylabel('Count', fontsize=12)
        axes[0, 0].set_title('Plate Length Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Plot 2: Aspect ratio distribution
        if len(high_conf) > 0:
            axes[0, 1].hist(high_conf['aspect_ratio'], bins=30, 
                           alpha=0.6, label='High Confidence', color='#2ecc71', edgecolor='black')
        if len(low_conf) > 0:
            axes[0, 1].hist(low_conf['aspect_ratio'], bins=30, 
                           alpha=0.6, label='Low Confidence', color='#e74c3c', edgecolor='black')
        axes[0, 1].set_xlabel('Aspect Ratio', fontsize=12)
        axes[0, 1].set_ylabel('Count', fontsize=12)
        axes[0, 1].set_title('Aspect Ratio Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Plot 3: BBox area distribution
        if len(high_conf) > 0:
            axes[1, 0].hist(high_conf['bbox_area'], bins=30, 
                           alpha=0.6, label='High Confidence', color='#2ecc71', edgecolor='black')
        if len(low_conf) > 0:
            axes[1, 0].hist(low_conf['bbox_area'], bins=30, 
                           alpha=0.6, label='Low Confidence', color='#e74c3c', edgecolor='black')
        axes[1, 0].set_xlabel('BBox Area (pixels²)', fontsize=12)
        axes[1, 0].set_ylabel('Count', fontsize=12)
        axes[1, 0].set_title('BBox Area Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Plot 4: Score vs Plate Length
        if len(high_conf) > 0:
            axes[1, 1].scatter(high_conf['plate_length'], high_conf['reliability_score'],
                              alpha=0.6, label='High Confidence', color='#2ecc71', 
                              edgecolors='black', s=50)
        if len(low_conf) > 0:
            axes[1, 1].scatter(low_conf['plate_length'], low_conf['reliability_score'],
                              alpha=0.6, label='Low Confidence', color='#e74c3c', 
                              edgecolors='black', s=50)
        axes[1, 1].set_xlabel('Plate Length', fontsize=12)
        axes[1, 1].set_ylabel('Reliability Score', fontsize=12)
        axes[1, 1].set_title('Score vs Plate Length', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'{output_dir}/deep_analysis.png', dpi=150, bbox_inches='tight')
        print(f"   ✅ Đã lưu biểu đồ: {output_dir}/deep_analysis.png")
        plt.show()
        
    except Exception as e:
        print(f"   ⚠️  Lỗi khi tạo biểu đồ: {e}")
    
    # Đề xuất hành động
    print("\n💡 ĐỀ XUẤT HÀNH ĐỘNG TỐI ƯU HÓA:")
    print("="*80)
    
    if all_issues:
        top_issue = issue_counts.most_common(1)[0]
        issue_name, issue_count = top_issue
        issue_pct = issue_count / len(df) * 100
        
        print(f"\n  🎯 VẤN ĐỀ CHÍNH: {issue_name} ({issue_pct:.1f}% samples)")
        
        if 'short' in issue_name.lower():
            print("\n  📋 HÀNH ĐỘNG:")
            print("    1. Tăng confidence threshold của YOLO (hiện tại: 0.20)")
            print("       → Thử: conf=0.30 hoặc conf=0.40")
            print("    2. Kiểm tra xem model có detect đúng license plates không")
            
        elif 'long' in issue_name.lower():
            print("\n  📋 HÀNH ĐỘNG:")
            print("    1. Giảm confidence threshold của YOLO")
            print("    2. Thêm post-processing: giới hạn độ dài tối đa")
            
        elif 'aspect' in issue_name.lower():
            print("\n  📋 HÀNH ĐỘNG:")
            print("    1. Thêm filter dựa trên aspect ratio trong post-processing")
            print("    2. License plates thường có aspect ratio 2.5-6.0")
            
        elif 'small' in issue_name.lower() or 'large' in issue_name.lower():
            print("\n  📋 HÀNH ĐỘNG:")
            print("    1. Thêm filter dựa trên diện tích bbox")
            print("    2. Loại bỏ các bbox quá nhỏ (< 1000 pixels²)")
    
    print("\n  🔧 CẢI THIỆN CHUNG:")
    print("    1. Thử nghiệm với các confidence thresholds khác nhau")
    print("    2. Thêm data augmentation cho PARSEQ")
    print("    3. Fine-tune YOLO với dataset cụ thể")
    print("    4. Thêm post-processing rules")
    
    print("="*80)
    
    return df


if __name__ == "__main__":
    df = deep_analysis('/kaggle/working/eval_no_gt')
