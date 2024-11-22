import torch
from torch.utils.data import DataLoader
from model_LSTM import LSTMClassifier
from train_LSTM import TextDataset, load_test_data, Vocabulary
import json
from tqdm import tqdm
import numpy as np

def load_label_map(label_file):
    """加载标签映射"""
    with open(label_file, 'r', encoding='utf-8') as f:
        labels_map = {line.strip(): idx for idx, line in enumerate(f)}
    return labels_map

def load_vocab_from_train_data(train_file):
    """从训练数据重建词典"""
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = [json.loads(line) for line in f]
    
    vocab = Vocabulary(min_freq=2)
    train_texts = [item['text'].lower() for item in train_data]
    vocab.build_vocab(train_texts)
    return vocab

def predict(model, test_loader, device, threshold=0.5):
    """
    对测试集进行预测
    """
    model.eval()
    all_predictions = []
    
    try:
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Testing'):
                text = batch['text'].to(device)
                length = batch['length']
                
                outputs = model(text, length)
                # 使用阈值进行多标签预测
                predictions = (outputs > threshold).float() # 将输出转换为0或1
                all_predictions.extend(predictions.cpu().numpy()) # 将预测结果转换为numpy数组
                
                # 清理内存
                del outputs, predictions
                if device == 'cuda':
                    torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise e
    
    return np.array(all_predictions)

def main():
    try:
        # 文件路径
        train_file = "data/train.json"
        test_file = "data/test.txt"
        label_file = "data/label_list.txt"
        model_path = "best_model_LSTM.pth"
        
        # 加载标签映射
        print("Loading label map...")
        labels_map = load_label_map(label_file)
        labels_reverse_map = {idx: label for label, idx in labels_map.items()}
        
        # 加载词典
        print("Loading vocabulary...")
        vocab = load_vocab_from_train_data(train_file)
        
        # 加载测试数据
        print("Loading test data...")
        test_data = load_test_data(test_file)
        
        # 创建测试数据集
        max_len = 128
        test_dataset = TextDataset(test_data, vocab, labels_map, max_len=max_len, is_test=True)
        
        # 创建数据加载器
        batch_size = 32
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # 创建模型并加载权重
        print("Loading model...")
        num_classes = len(labels_map)
        model = LSTMClassifier(
            vocab_size=len(vocab),
            embedding_dim=300,
            hidden_dim=256,
            num_classes=num_classes
        )
        
        # 加载模型权重
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']} with F1 score {checkpoint['val_f1']:.4f}")
        
        model = model.to(device)
        
        # 进行预测
        print("Starting prediction...")
        predictions = predict(model, test_loader, device)
        
        # 将预测结果转换为标签
        predicted_labels = []
        for pred in predictions:
            # 获取所有预测为1的类别
            labels = [labels_reverse_map[idx] for idx, value in enumerate(pred) if value == 1]
            # 如果没有预测为1的类别，选择概率最大的类别
            if not labels:
                max_idx = np.argmax(pred) # 获取概率最大的类别
                labels = [labels_reverse_map[max_idx]] # 将概率最大的类别添加到预测标签中
            predicted_labels.append(labels) # 将预测标签添加到预测标签列表中
        
        # 保存预测结果
        print("Saving predictions...")
        with open('predictions_LSTM.txt', 'w', encoding='utf-8') as f:
            for labels in predicted_labels:
                # 将多个标签用逗号连接
                f.write(f"{','.join(labels)}\n")
        
        print("Testing completed!")
        print(f"Total predictions: {len(predicted_labels)}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise e

if __name__ == "__main__":
    main()
