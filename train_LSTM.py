import torch  
from torch.utils.data import Dataset, DataLoader  
import torch.nn as nn  
import torch.nn.functional as F  
from model_LSTM import LSTMClassifier  # 导入自定义的LSTM分类器模型
import json  # JSON数据处理
from collections import Counter  # 用于词频统计
import numpy as np  
from tqdm import tqdm  # 进度条显示
from sklearn.metrics import f1_score  # F1评分计算
import os  

def load_data(train_file, val_file, label_file):
    """
    加载训练集、验证集和标签文件
    Args:
        train_file: 训练数据文件路径
        val_file: 验证数据文件路径
        label_file: 标签映射文件路径
    Returns:
        训练数据、验证数据和标签映射字典
    """
    # 读取标签映射文件，创建标签到索引的映射
    with open(label_file, 'r', encoding='utf-8') as f:
        labels_map = {line.strip(): idx for idx, line in enumerate(f)}
    
    # 读取训练数据
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = [json.loads(line) for line in f]
    
    # 读取验证数据
    with open(val_file, 'r', encoding='utf-8') as f:
        val_data = [json.loads(line) for line in f]
    
    return train_data, val_data, labels_map

def load_test_data(test_file):
    """
    加载测试集文件
    """
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = [line.strip() for line in f]
    return test_data

class Vocabulary:
    """
    词汇表类，用于构建词汇到索引的映射
    """
    def __init__(self, min_freq=1):
        self.min_freq = min_freq  # 最小词频阈值
        # 初始化特殊标记
        """
        <pad>：填充标记，用于处理序列长度不一致的情况
        <unk>：未知词标记，用于处理词汇表中不存在的词
        """
        self.word2idx = {'<pad>': 0, '<unk>': 1}  # 词到索引的映射，将词汇转化为整数索引
        self.idx2word = {0: '<pad>', 1: '<unk>'}  # 索引到词的映射，将整数索引转化为词汇
        self.word_freq = Counter()  # 词频统计
        
    def build_vocab(self, texts):
        """构建词汇表"""
        # 统计词频
        for text in texts:
            self.word_freq.update(text.split())
        
        # 为频率大于等于min_freq的词分配索引   
        # 索引和词汇一一对应 
        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
                
    def __len__(self):
        """返回词汇表大小"""
        return len(self.word2idx)
    
    def text_to_indices(self, text, max_len=None):
        """
        将文本转换为索引序列
        Args:
            text: 输入文本
            max_len: 最大序列长度
        Returns:
            索引序列
        """
        # 将文本转换为索引序列，对于未知词使用<unk>的索引
        indices = [self.word2idx.get(word, self.word2idx['<unk>']) for word in text.split()]
        if max_len is not None:
            # 处理序列长度：截断或填充
            if len(indices) < max_len:
                indices = indices + [self.word2idx['<pad>']] * (max_len - len(indices)) # 不足max_len则填充
            else:
                indices = indices[:max_len] # 超过max_len则截断
        return indices

class TextDataset(Dataset):
    """
    文本数据集类，用于加载和预处理文本数据
    """
    def __init__(self, data, vocab, labels_map, max_len=128, is_test=False):
        self.vocab = vocab
        self.max_len = max_len
        self.is_test = is_test # 是否为测试数据
        
        if not is_test: 
            # 处理训练和验证数据
            self.texts = [item['text'] for item in data]
            # 处理多标签情况
            self.labels = []
            for item in data:
                # 创建标签的one-hot向量
                label_vector = torch.zeros(len(labels_map))
                # 处理单标签和多标签的情况
                if isinstance(item['label'], list): # 多标签
                    for label in item['label']:
                        if label in labels_map:
                            label_vector[labels_map[label]] = 1
                else: # 单标签
                    if item['label'] in labels_map:
                        label_vector[labels_map[item['label']]] = 1
                self.labels.append(label_vector)
        else:
            # 处理测试数据
            self.texts = data if isinstance(data, list) else [data]
            self.labels = None

    def __len__(self):
        """返回数据集大小"""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """获取数据集中的一个样本"""
        # 转换文本为小写并获取索引序列
        text = str(self.texts[idx]).lower()
        indices = self.vocab.text_to_indices(text, self.max_len)
        length = min(len(text.split()), self.max_len)   # 取单词列表text.split()和max_len的最小值
        
        # 构建返回字典
        item = {
            'text': torch.tensor(indices, dtype=torch.long),  # 将索引序列转换为张量
            'length': torch.tensor(length, dtype=torch.long)  # 将长度转换为张量
        }
        
        if not self.is_test:
            item['labels'] = self.labels[idx]  # 将标签添加到字典中
        return item

def train_model(model, train_loader, val_loader, epochs=10, device='cuda'):
    """
    训练模型
    Args:
        model: LSTM模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        device: 训练设备
    """
    # 初始化优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters()) # Adam优化器
    criterion = nn.BCELoss()  # 二元交叉熵损失函数
    
    # 将模型移到指定设备
    model = model.to(device)
    best_val_f1 = 0
    
    # 开始训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        # 处理每个批次的数据
        for batch in train_pbar:
            # 准备数据
            text = batch['text'].to(device)
            length = batch['length']
            labels = batch['labels'].float().to(device)
            
            # 前向传播和反向传播
            optimizer.zero_grad() # 梯度清零
            outputs = model(text, length) # 前向传播，获取模型输出
            loss = criterion(outputs, labels) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 更新模型参数
            
            total_loss += loss.item() # 累加损失
            
            # 计算训练准确率
            predictions = (outputs > 0.5).float()
            correct = (predictions == labels).float() # 计算预测结果与真实标签的准确率
            accuracy = correct.sum(dim=1) / labels.size(1) # 计算每个样本的准确率
            train_correct += accuracy.sum().item() # 累加准确率
            train_total += labels.size(0) # 累加样本数量
            
            # 更新进度条信息
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_correct/train_total:.4f}'
            })
            
            # 清理内存
            del outputs, loss, predictions, correct, accuracy
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        # 计算平均训练准确率
        train_accuracy = train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        print("\nValidating...")
        with torch.no_grad():  # 禁用梯度计算，好处是节省内存
            for batch in tqdm(val_loader, desc='Validation'): # 使用tqdm显示进度条
                # 准备数据
                text = batch['text'].to(device)
                length = batch['length']
                labels = batch['labels'].float().to(device)
                
                # 前向传播
                outputs = model(text, length) # 获取模型输出
                loss = criterion(outputs, labels) # 计算损失
                val_loss += loss.item() # 累加损失
                
                # 计算预测结果
                predictions = (outputs > 0.5).float() # 将输出转换为0或1
                
                # 计算验证准确率
                correct = (predictions == labels).float()
                accuracy = correct.sum(dim=1) / labels.size(1)
                val_correct += accuracy.sum().item()
                val_total += labels.size(0)
                
                # 收集预测结果用于计算F1分数
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # 清理内存
                del outputs, predictions, labels, loss, correct, accuracy
                if device == 'cuda':
                    torch.cuda.empty_cache()
        
        # 计算验证指标
        val_accuracy = val_correct / val_total
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # 计算每个类别的F1分数
        f1_scores = []
        for i in range(all_preds.shape[1]):
            f1 = f1_score(all_labels[:, i], all_preds[:, i], zero_division=1)
            f1_scores.append(f1)
        
        avg_f1 = np.mean(f1_scores)
        
        # 打印训练信息
        print(f'Epoch {epoch+1}:')
        print(f'Average training loss: {total_loss/len(train_loader):.4f}')
        print(f'Training accuracy: {train_accuracy:.4f}')
        print(f'Average validation loss: {val_loss/len(val_loader):.4f}')
        print(f'Validation accuracy: {val_accuracy:.4f}')
        print(f'Average F1 score: {avg_f1:.4f}')
        
        # 保存最佳模型
        if avg_f1 > best_val_f1:
            best_val_f1 = avg_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(), # 保存模型参数
                'optimizer_state_dict': optimizer.state_dict(), # 保存优化器参数
                'val_f1': avg_f1, # 保存验证F1分数
                'val_accuracy': val_accuracy, # 保存验证准确率
            }, 'best_model_LSTM.pth')
            print(f"Saved new best model with F1 score: {avg_f1:.4f} and accuracy: {val_accuracy:.4f}")

# 主程序入口
if __name__ == "__main__":
    # 定义文件路径
    train_file = "data/train.json"
    val_file = "data/valid.json"
    test_file = "data/test.txt"
    label_file = "data/label_list.txt"
        
    # 检查文件是否存在
    for file_path in [train_file, val_file, test_file, label_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
    print("Loading data...")
    # 加载数据
    train_data, val_data, labels_map = load_data(train_file, val_file, label_file)
    test_data = load_test_data(test_file)
        
    print("Building vocabulary...")
    # 构建词典
    vocab = Vocabulary(min_freq=2)
    train_texts = [item['text'].lower() for item in train_data]
    vocab.build_vocab(train_texts)
        
    # 设置模型参数
    vocab_size = len(vocab) # 词汇表大小
    embedding_dim = 300  # 词嵌入维度
    hidden_dim = 256    # LSTM隐藏层维度
    num_classes = len(labels_map)  # 分类类别数
        
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of classes: {num_classes}")
        
    # 创建LSTM模型
    model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, num_classes)
        
    print("Creating datasets...")
    # 创建数据集
    train_dataset = TextDataset(train_data, vocab, labels_map)
    val_dataset = TextDataset(val_data, vocab, labels_map)
    test_dataset = TextDataset(test_data, vocab, labels_map, is_test=True)
        
    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
    # 设置计算设备（GPU/CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
        
    # 开始训练模型
    print("Starting training...")
    train_model(model, train_loader, val_loader, device=device)
        
    print("Training completed!")
        
    
