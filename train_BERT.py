import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F
from model_BERT import BertClassifier
import json
import gc
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, data, labels_map, tokenizer, max_len=128, is_test=False):
        self.tokenizer = tokenizer # 分词器
        self.max_len = max_len # 最大序列长度
        self.is_test = is_test # 是否为测试集
        
        if not is_test:
            self.texts = [item['text'] for item in data]
            self.labels = []
            for item in data:
                label = item['label'][0] if isinstance(item['label'], list) else item['label'] # 获取标签
                self.labels.append(labels_map[label]) # 将标签转换为索引
        else:
            self.texts = data
            self.labels = None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]).lower() # 将文本转换为小写
        encoding = self.tokenizer.encode_plus( # 编码文本
            text,
            add_special_tokens=True, # 添加特殊标记，如[CLS]和[SEP]，用于标识句子开头和结尾
            max_length=self.max_len, # 最大序列长度
            padding='max_length', # 填充
            truncation=True, # 截断
            return_attention_mask=True, # 返回注意力掩码，标记哪些位置时有效的输入，哪些位置是填充的位置
            return_tensors='pt' # 返回PyTorch张量
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(), # 展平输入ID
            'attention_mask': encoding['attention_mask'].flatten(), # 展平注意力掩码
        }
        
        if not self.is_test:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def load_data(train_file, val_file, label_file):
    with open(label_file, 'r', encoding='utf-8') as f:
        labels_map = {line.strip(): idx for idx, line in enumerate(f)}
    
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = [json.loads(line) for line in f]
    
    with open(val_file, 'r', encoding='utf-8') as f:
        val_data = [json.loads(line) for line in f]
    
    return train_data, val_data, labels_map

def load_test_data(test_file):
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = [line.strip() for line in f]
    return test_data

def train_model(model, train_loader, val_loader, epochs=3, device='cuda'):
    # 优化器设置
    #model.parameters()获取模型中所需要训练的参数，以供优化器更新，是一个可迭代的对象
    #设置学习率lr
    #设置权重衰减系数，用于L2正则化，目的：限制权重大小，避免模型过拟合
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01) # AdamW是Adam优化器的改进版本
    
    # 计算总训练步数
    total_steps = len(train_loader) * epochs
    
    # 创建学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,  # 10% 的步数用于预热
        num_training_steps=total_steps
    )
    
    # 设置梯度累积步数
    gradient_accumulation_steps = 4
    
    # 将模型移至GPU
    model = model.to(device)
    
    # 启用混合精度训练，目的：解决gpu内存不足的问题
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        
        # 训练阶段
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        train_pbar = tqdm(train_loader, desc='Training')
        for batch_idx, batch in enumerate(train_pbar):
            try:
                # 清理缓存
                if device == 'cuda':
                    torch.cuda.empty_cache()
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # 使用混合精度训练
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = F.cross_entropy(outputs, labels)
                    loss = loss / gradient_accumulation_steps #每4步后再执行依次优化器的更新，减少了对每批次gpu内存的需求
                
                # 使用scaler进行反向传播
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # 优化器步进
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * gradient_accumulation_steps
                
                # 更新进度条
                train_pbar.set_postfix({'loss': loss.item()})
                
                # 释放内存
                del outputs, loss
                if device == 'cuda':
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                print(f"Batch {batch_idx} failed, skipping...")
                print(e)
                continue
        
        avg_train_loss = total_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_acc = 0
        val_count = 0
        
        print("\nValidating...")
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs, dim=1)
                val_acc += (predictions == labels).sum().item()
                val_count += labels.size(0)
                
                del outputs, predictions
                if device == 'cuda':
                    torch.cuda.empty_cache()
        
        val_accuracy = val_acc / val_count
        print(f'Epoch {epoch+1}:')
        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Validation accuracy: {val_accuracy:.4f}')
        
        # 保存最佳模型
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            # 保存完整模型状态
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
            }, 'best_model_BERT.pth')
            print(f"Saved new best model with validation accuracy: {val_accuracy:.4f}")
        
        # 在第2个epoch结束后解冻所有BERT层
        if epoch == 1:
            print("Unfreezing all BERT layers...")
            model.unfreeze_bert_layers()
            
        # 垃圾回收
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        # 设置环境变量
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        # 文件路径
        train_file = "data/train.json"
        val_file = "data/valid.json"
        test_file = "data/test.txt"
        label_file = "data/label_list.txt"
        
        print("Loading data...")
        train_data, val_data, labels_map = load_data(train_file, val_file, label_file)
        test_data = load_test_data(test_file)
        
        # 初始化tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # 创建数据集
        max_len = 128  # 设置合适的序列长度
        train_dataset = TextDataset(train_data, labels_map, tokenizer, max_len=max_len)
        val_dataset = TextDataset(val_data, labels_map, tokenizer, max_len=max_len)
        test_dataset = TextDataset(test_data, labels_map, tokenizer, max_len=max_len, is_test=True)
        
        # 创建数据加载器
        batch_size = 8  # 根据GPU内存调整
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # 创建模型
        num_classes = len(labels_map)
        model = BertClassifier(num_classes=num_classes)
        
        # 训练模型
        print("Starting training...")
        train_model(model, train_loader, val_loader, device=device)
        
        print("Training completed!")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise e
