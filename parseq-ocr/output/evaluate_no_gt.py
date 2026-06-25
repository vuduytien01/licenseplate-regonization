import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

def evaluate_ocr_no_gt(csv_path, image_dir, output_dir='/kaggle/working/eval_no_gt'):
    """
    Đánh giá chất lượng OCR predictions mà không cần ground truth.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("🚀 BẮT ĐẦU ĐÁNH GIÁ OCR (KHÔNG CẦN GROUND TRUTH)")
    print("="*80)
    
    # Load predictions
    print(f"\n📂 Loading predictions từ: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"   ✅ Loaded {len(df)} predictions")
    except Exception as e:
        print(f"   ❌ Lỗi khi load file: {e}")
        return None
    
    # Parse bbox
    def parse_bbox(bbox_str):
        try:
            if isinstance(bbox_str, str):
                bbox_str = bbox_str.strip()
                if bbox_str.startswith('[') and bbox_str.endswith(']'):
                    return eval(bbox_str)
                elif bbox_str.startswith('(') and bbox_str.endswith(')'):
                    return eval(bbox_str)
            elif isinstance(bbox_str, (list, tuple)):
                return list(bbox_str)
            return None
        except:
            return None
    
    df['bbox_parsed'] = df['bbox'].apply(parse_bbox)
    valid_mask = df['bbox_parsed'].apply(lambda x: x is not None and len(x) == 4)
    df = df[valid_mask].copy()
    
    print(f"   ✅ {len(df)} predictions có bbox hợp lệ")
    
    if len(df) == 0:
        print("   ❌ Không có predictions hợp lệ!")
        return None
    
    # Tính features
    df['bbox_width'] = df['bbox_parsed'].apply(lambda x: abs(x[2] - x[0]))
    df['bbox_height'] = df['bbox_parsed'].apply(lambda x: abs(x[3] - x[1]))
    df['bbox_area'] = df['bbox_width'] * df['bbox_height']
    df['aspect_ratio'] = df['bbox_width'] / (df['bbox_height'] + 1e-6)
    df['plate'] = df['plate'].astype(str).str.upper().str.strip()
    df['plate_length'] = df['plate'].str.len()
    
    # Tính reliability score
    print("\n🔢 Đang tính reliability score...")
    
    reliability_scores = []
    
    for idx, row in df.iterrows():
        score = 100
        issues = []
        
        plate = row['plate']
        
        # Kiểm tra độ dài
        if row['plate_length'] < 3:
            score -= 30
            issues.append('Too short')
        elif row['plate_length'] > 10:
            score -= 20
            issues.append('Too long')
        elif 6 <= row['plate_length'] <= 9:
            score += 10
            issues.append('Good length')
        
        # Kiểm tra ký tự
        if not re.match(r'^[A-Z0-9]+$', plate):
            score -= 40
            issues.append('Invalid characters')
        
        # Kiểm tra aspect ratio
        if row['aspect_ratio'] < 2:
            score -= 20
            issues.append('Too square')
        elif row['aspect_ratio'] > 7:
            score -= 15
            issues.append('Too elongated')
        elif 3 <= row['aspect_ratio'] <= 5:
            score += 10
            issues.append('Good aspect ratio')
        
        # Kiểm tra diện tích
        if row['bbox_area'] < 1000:
            score -= 25
            issues.append('Very small')
        elif row['bbox_area'] > 50000:
            score -= 10
            issues.append('Very large')
        
        reliability_scores.append({
            'image': row['image'],
            'plate': plate,
            'reliability_score': max(0, score),
            'issues': ', '.join(issues) if issues else 'None',
            'bbox_area': row['bbox_area'],
            'aspect_ratio': row['aspect_ratio'],
            'plate_length': row['plate_length'],
            'bbox': row['bbox_parsed']
        })
    
    df_result = pd.DataFrame(reliability_scores)
    
    print(f"   ✅ Đã tính score cho {len(df_result)} predictions")
    
    # Thống kê
    print("\n" + "="*80)
    print("📊 KẾT QUẢ ĐÁNH GIÁ")
    print("="*80)
    
    avg_score = df_result['reliability_score'].mean()
    median_score = df_result['reliability_score'].median()
    
    high_conf = df_result[df_result['reliability_score'] >= 80]
    medium_conf = df_result[(df_result['reliability_score'] >= 60) & 
                           (df_result['reliability_score'] < 80)]
    low_conf = df_result[df_result['reliability_score'] < 60]
    
    print(f"\n📈 TỔNG QUAN:")
    print(f"  • Tổng số predictions: {len(df_result)}")
    print(f"  • Điểm trung bình: {avg_score:.1f}/100")
    print(f"  • Điểm trung vị: {median_score:.1f}/100")
    
    print(f"\n🎯 PHÂN LOẠI ĐỘ TIN CẬY:")
    print(f"  • High confidence (≥80): {len(high_conf):4d} ({len(high_conf)/len(df_result)*100:.1f}%)")
    print(f"  • Medium confidence (60-79): {len(medium_conf):4d} ({len(medium_conf)/len(df_result)*100:.1f}%)")
    print(f"  • Low confidence (<60): {len(low_conf):4d} ({len(low_conf)/len(df_result)*100:.1f}%)")
    
    # Phân tích issues
    all_issues = []
    for issues in df_result['issues']:
        if issues != 'None':
            all_issues.extend([i.strip() for i in issues.split(',')])
    
    if all_issues:
        issue_counts = Counter(all_issues)
        print(f"\n⚠️  CÁC VẤN ĐỀ PHỔ BIẾN:")
        for issue, count in issue_counts.most_common(5):
            print(f"    • {issue:30s}: {count:3d} lần")
    
    print("="*80)
    
    # Trực quan hóa
    print("\n📊 Đang tạo biểu đồ...")
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('🔍 OCR Quality Assessment (Without Ground Truth)', 
                     fontsize=18, fontweight='bold', y=0.998)
        
        # Plot 1: Distribution of reliability score
        axes[0, 0].hist(df_result['reliability_score'], bins=20, 
                       color='#2ecc71', edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Reliability Score', fontsize=12)
        axes[0, 0].set_ylabel('Count', fontsize=12)
        axes[0, 0].set_title('Distribution of Reliability Scores', 
                            fontsize=14, fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].axvline(70, color='red', linestyle='--', label='Threshold: 70')
        axes[0, 0].legend()
        
        # Plot 2: Score vs Plate Length
        axes[0, 1].scatter(df_result['plate_length'], 
                          df_result['reliability_score'],
                          alpha=0.6, c='#3498db', edgecolors='black', s=50)
        axes[0, 1].set_xlabel('Plate Length', fontsize=12)
        axes[0, 1].set_ylabel('Reliability Score', fontsize=12)
        axes[0, 1].set_title('Reliability vs Plate Length', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)
        
        # Plot 3: Score vs Aspect Ratio
        axes[0, 2].scatter(df_result['aspect_ratio'], 
                          df_result['reliability_score'],
                          alpha=0.6, c='#e74c3c', edgecolors='black', s=50)
        axes[0, 2].set_xlabel('Aspect Ratio', fontsize=12)
        axes[0, 2].set_ylabel('Reliability Score', fontsize=12)
        axes[0, 2].set_title('Reliability vs Aspect Ratio', 
                            fontsize=14, fontweight='bold')
        axes[0, 2].grid(alpha=0.3)
        
        # Plot 4: Issues distribution
        if all_issues:
            issue_labels = list(issue_counts.keys())[:8]
            issue_values = list(issue_counts.values())[:8]
            
            axes[1, 0].barh(issue_labels[::-1], issue_values[::-1],
                           color='#f39c12', edgecolor='black', alpha=0.8)
            axes[1, 0].set_xlabel('Count', fontsize=12)
            axes[1, 0].set_title('Top Issues Detected', 
                                fontsize=14, fontweight='bold')
            axes[1, 0].grid(axis='x', alpha=0.3)
        
        # Plot 5: Plate length distribution
        axes[1, 1].hist(df_result['plate_length'], bins=range(1, 15),
                       color='#9b59b6', edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Plate Length', fontsize=12)
        axes[1, 1].set_ylabel('Count', fontsize=12)
        axes[1, 1].set_title('Plate Length Distribution', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # Plot 6: High vs Low confidence
        axes[1, 2].pie([len(high_conf), len(medium_conf), len(low_conf)], 
                      labels=[f'High (≥80)\n{len(high_conf)}', 
                             f'Medium (60-79)\n{len(medium_conf)}', 
                             f'Low (<60)\n{len(low_conf)}'],
                      colors=['#2ecc71', '#f39c12', '#e74c3c'],
                      autopct='%1.1f%%', startangle=90)
        axes[1, 2].set_title('Confidence Distribution', 
                            fontsize=14, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.995])
        plt.savefig(f'{output_dir}/quality_assessment.png', dpi=150, bbox_inches='tight')
        print(f"   ✅ Đã lưu biểu đồ: {output_dir}/quality_assessment.png")
        plt.show()
        
    except Exception as e:
        print(f"   ⚠️  Lỗi khi tạo biểu đồ: {e}")
    
    # Export kết quả
    print("\n💾 Đang export kết quả...")
    
    reliability_csv = f'{output_dir}/reliability_assessment.csv'
    df_result.to_csv(reliability_csv, index=False)
    print(f"   ✅ File đánh giá: {reliability_csv}")
    
    suspicious = df_result[df_result['reliability_score'] < 50]
    if len(suspicious) > 0:
        suspicious_csv = f'{output_dir}/suspicious_predictions.csv'
        suspicious.to_csv(suspicious_csv, index=False)
        print(f"   ✅ Predictions đáng ngờ (score < 50): {suspicious_csv}")
        print(f"      → {len(suspicious)} predictions")
    
    high_quality = df_result[df_result['reliability_score'] >= 80]
    if len(high_quality) > 0:
        high_quality_csv = f'{output_dir}/high_quality_predictions.csv'
        high_quality.to_csv(high_quality_csv, index=False)
        print(f"   ✅ Predictions chất lượng cao (score ≥ 80): {high_quality_csv}")
        print(f"      → {len(high_quality)} predictions")
    
    print("\n" + "="*80)
    print("✅ HOÀN TẤT ĐÁNH GIÁ!")
    print(f"📁 Tất cả kết quả đã lưu tại: {output_dir}")
    print("="*80)
    
    return df_result


if __name__ == "__main__":
    CSV_PATH = "/kaggle/working/license_plate_results.csv"
    IMAGE_DIR = "/kaggle/input/datasets/vdt1501/license-plate-detection-dataset/License Plate Detection Dataset/images/test"
    OUTPUT_DIR = "/kaggle/working/eval_no_gt"
    
    df_result = evaluate_ocr_no_gt(CSV_PATH, IMAGE_DIR, OUTPUT_DIR)
