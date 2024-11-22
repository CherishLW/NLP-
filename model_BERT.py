import torch
import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, num_classes, pretrained_model="bert-base-uncased"):
        super(BertClassifier, self).__init__()
        # 初始化BERT模型
        self.bert = BertModel.from_pretrained(pretrained_model)
        
        # 启用梯度检查点以节省内存
        self.bert.gradient_checkpointing_enable()
        
        # 配置dropout和分类器
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        # 冻结部分BERT层以减少内存使用
        modules_to_freeze = [
            self.bert.embeddings,
            *self.bert.encoder.layer[:8]  # 冻结前8层
        ]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        # 使用no_grad来减少不必要的梯度计算
        with torch.set_grad_enabled(self.training):
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,  # 不输出所有隐藏层状态
                return_dict=False  # 返回元组而不是字典以节省内存
            )
            
            pooled_output = outputs[1]  # 获取[CLS]标记的输出
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            
            return logits
    
    def unfreeze_bert_layers(self):
        """
        训练过程中可以调用此方法来逐步解冻BERT层
        """
        for param in self.bert.parameters():
            param.requires_grad = True
