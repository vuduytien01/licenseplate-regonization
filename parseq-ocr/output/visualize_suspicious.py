import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

def visualize_suspicious_predictions(output_dir='/kaggle/working/eval_no_gt', 
                                    image_dir="/kaggle/input/datasets/vdt1501/license-plate-detection-dataset/License Plate Detection Dataset/images/test"):
    """Trực quan hóa các predictions đáng ngờ."""
    
    suspicious_csv = f'{output_dir}/suspicious_predictions.csv'
    
    if not os.path.exists(suspicious_csv):
        print("❌ Không tìm thấy file suspicious_predictions.csv")
        print("   Có thể không có predictions nào có score < 50")
        return
    
    df = pd.read_csv(suspicious_csv)
    
    if len(df) == 0:
        print("✅ Không có predictions đáng ngờ!")
        return
    
    print(f"🔍 Đang trực quan hóa {len(df)} predictions đáng ngờ...")
    
    # Tạo thư mục output
    vis_dir = f'{output_dir}/suspicious_visualizations'
    os.makedirs(vis_dir, exist_ok=True)
    
    # Hiển thị top 20 predictions đáng ngờ nhất
    top_suspicious = df.nsmallest(min(20, len(df)), 'reliability_score')
    
    n_rows = (len(top_suspicious) + 4) // 5  # 5 cột
    n_cols = min(5, len(top_suspicious))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    fig.suptitle('🚨 Top Suspicious Predictions', 
                 fontsize=18, fontweight='bold', y=0.998)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(n_rows * n_cols):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        
        ax = axes[row_idx, col_idx]
        
        if idx < len(top_suspicious):
            row = top_suspicious.iloc[idx]
            
            img_path = os.path.join(image_dir, row['image'])
            
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Parse bbox
                try:
                    bbox_str = str(row['bbox'])
                    if bbox_str.startswith('['):
                        bbox = eval(bbox_str)
                    else:
                        bbox = [float(x) for x in bbox_str.strip('[]()').split(',')]
                    
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                        
                        cv2.putText(img, f"Score: {row['reliability_score']:.0f}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                        cv2.putText(img, f"Pred: {row['plate']}", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                except:
                    pass
                
                ax.imshow(img)
                ax.set_title(f"{row['image'][:25]}...\nScore: {row['reliability_score']:.0f} | "
                            f"Plate: '{row['plate']}'",
                            fontsize=9, fontweight='bold', color='red')
            else:
                ax.text(0.5, 0.5, f"Image not found\n{row['image']}",
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, color='red')
        else:
            ax.axis('off')
        
        ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.995])
    plt.savefig(f'{vis_dir}/suspicious_predictions.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Đã lưu visualization tại: {vis_dir}/suspicious_predictions.png")
    plt.show()
    
    # Lưu từng ảnh riêng biệt
    print(f"\n💾 Đang lưu từng ảnh riêng biệt...")
    saved_count = 0
    for idx, row in top_suspicious.iterrows():
        img_path = os.path.join(image_dir, row['image'])
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            
            try:
                bbox_str = str(row['bbox'])
                if bbox_str.startswith('['):
                    bbox = eval(bbox_str)
                else:
                    bbox = [float(x) for x in bbox_str.strip('[]()').split(',')]
                
                if len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    
                    cv2.putText(img, f"Score: {row['reliability_score']:.0f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(img, f"Pred: {row['plate']}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    output_path = f"{vis_dir}/{row['image']}"
                    cv2.imwrite(output_path, img)
                    saved_count += 1
            except:
                pass
    
    print(f"✅ Đã lưu {saved_count} ảnh tại: {vis_dir}/")


if __name__ == "__main__":
    visualize_suspicious_predictions()
