import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model_BERT import BertClassifier
from train_BERT import TextDataset, load_test_data
import json
from tqdm import tqdm

def load_label_map(label_file):
    """加载标签映射"""
    with open(label_file, 'r', encoding='utf-8') as f:
        labels_map = {line.strip(): idx for idx, line in enumerate(f)}
    return labels_map

def predict(model, test_loader, device):
    """
    对测试集进行预测
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            
            # 清理内存
            del outputs, preds
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    return predictions

def main():
    try:
        # 文件路径
        test_file = "data/test.txt"
        label_file = "data/label_list.txt"
        model_path = "best_model_BERT.pth"  # 最佳模型的保存路径
        
        # 加载标签映射
        labels_map = load_label_map(label_file)
        labels_reverse_map = {idx: label for label, idx in labels_map.items()}
        
        # 加载测试数据
        print("Loading test data...")
        test_data = load_test_data(test_file)
        
        # 初始化tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # 创建测试数据集
        max_len = 128  # 保持与训练时相同的长度
        test_dataset = TextDataset(test_data, labels_map, tokenizer, max_len=max_len, is_test=True)
        
        # 创建数据加载器
        batch_size = 8
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # 创建模型并加载权重
        print("Loading model...")
        num_classes = len(labels_map)
        model = BertClassifier(num_classes=num_classes)
        
        # 加载模型权重
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']} with validation accuracy {checkpoint['val_accuracy']:.4f}")
        
        model = model.to(device)
        
        # 进行预测
        print("Starting prediction...")
        predictions = predict(model, test_loader, device)
        
        # 将预测结果转换为标签
        predicted_labels = [labels_reverse_map[pred] for pred in predictions]
        
        # 保存预测结果
        print("Saving predictions...")
        with open('predictions_BERT.txt', 'w', encoding='utf-8') as f:
            for label in predicted_labels:
                f.write(f"{label}\n")
        
        print("Testing completed!")
        print(f"Total predictions: {len(predicted_labels)}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise e

if __name__ == "__main__":
    main()
