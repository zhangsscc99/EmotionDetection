"""
直接生成混淆矩阵和评估报告的脚本
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import create_data_generators
from config import TEST_DIR, EMOTION_LABELS

def main():
    # 确保输出目录存在
    os.makedirs("plots", exist_ok=True)
    
    print("加载模型...")
    model_path = "models/saved/emotion_model.h5"
    
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        print("请确认模型已训练并保存在正确位置")
        return
    
    try:
        model = tf.keras.models.load_model(model_path)
        print("模型加载成功!")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return
    
    print("加载测试数据...")
    try:
        _, _, test_generator = create_data_generators(TEST_DIR, TEST_DIR)
        print(f"测试数据加载成功! 共 {test_generator.samples} 个样本")
    except Exception as e:
        print(f"加载测试数据时出错: {e}")
        return
    
    print("评估模型性能...")
    try:
        # 评估模型
        test_loss, test_accuracy = model.evaluate(test_generator)
        print(f"测试损失: {test_loss:.4f}")
        print(f"测试准确率: {test_accuracy:.4f}")
        
        # 获取预测
        print("获取预测结果...")
        y_pred_prob = model.predict(test_generator)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = test_generator.classes
        
        # 生成分类报告
        print("\n分类报告:")
        class_report = classification_report(
            y_true, y_pred,
            target_names=list(EMOTION_LABELS.values())
        )
        print(class_report)
        
        # 保存分类报告到文件
        with open("classification_report.txt", "w") as f:
            f.write(f"测试损失: {test_loss:.4f}\n")
            f.write(f"测试准确率: {test_accuracy:.4f}\n\n")
            f.write(class_report)
        print("分类报告已保存到 classification_report.txt")
        
        # 生成混淆矩阵
        print("生成混淆矩阵...")
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # 计算归一化混淆矩阵（按行归一化）
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        # 可视化混淆矩阵
        plt.figure(figsize=(14, 12))
        
        # 1. 原始混淆矩阵
        plt.subplot(1, 2, 1)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(EMOTION_LABELS.values()),
                   yticklabels=list(EMOTION_LABELS.values()))
        plt.title('混淆矩阵', fontsize=16)
        plt.xlabel('预测标签', fontsize=14)
        plt.ylabel('真实标签', fontsize=14)
        
        # 2. 归一化混淆矩阵
        plt.subplot(1, 2, 2)
        sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=list(EMOTION_LABELS.values()),
                   yticklabels=list(EMOTION_LABELS.values()))
        plt.title('归一化混淆矩阵', fontsize=16)
        plt.xlabel('预测标签', fontsize=14)
        plt.ylabel('真实标签', fontsize=14)
        
        plt.tight_layout()
        plt.savefig('plots/confusion_matrix.png', dpi=300)
        print("混淆矩阵已保存为 plots/confusion_matrix.png")
        
        # 计算每个类别的准确率
        class_accuracy = {}
        for i, emotion in EMOTION_LABELS.items():
            class_mask = (y_true == i)
            if np.sum(class_mask) > 0:  # 避免除以零
                class_acc = np.sum((y_pred == i) & class_mask) / np.sum(class_mask)
                class_accuracy[emotion] = class_acc
        
        # 可视化每个类别的准确率
        plt.figure(figsize=(12, 6))
        emotions = list(class_accuracy.keys())
        accuracies = list(class_accuracy.values())
        
        bars = plt.bar(emotions, accuracies, color='skyblue')
        plt.title('各情绪类别的准确率', fontsize=16)
        plt.xlabel('情绪类别', fontsize=14)
        plt.ylabel('准确率', fontsize=14)
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 在柱状图上添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('plots/class_accuracy.png', dpi=300)
        print("类别准确率图已保存为 plots/class_accuracy.png")
        
        # 可视化模型的ROC曲线
        try:
            from sklearn.metrics import roc_curve, auc
            from sklearn.preprocessing import label_binarize
            
            # 将标签二值化
            y_true_bin = label_binarize(y_true, classes=range(len(EMOTION_LABELS)))
            
            plt.figure(figsize=(10, 8))
            
            for i, emotion in EMOTION_LABELS.items():
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{emotion} (AUC = {roc_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假阳性率', fontsize=14)
            plt.ylabel('真阳性率', fontsize=14)
            plt.title('各情绪类别的ROC曲线', fontsize=16)
            plt.legend(loc="lower right", fontsize=12)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('plots/roc_curves.png', dpi=300)
            print("ROC曲线已保存为 plots/roc_curves.png")
        except Exception as e:
            print(f"生成ROC曲线时出错: {e}")
        
    except Exception as e:
        print(f"评估模型时出错: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n评估和可视化完成!")

if __name__ == "__main__":
    main() 