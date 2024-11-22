# NLP-期末大作业
本项目使用了两种方法来实现隐私文本分类问题，分别是LSTM和BERT。
LSTM准确率达到98%左右，实现了多标签分类问题。
BERT准确率达到88%左右，实现了多分类问题，但预测结果的标签仍然是单标签。

LSTM：
训练：运行train_LSTM.py
测试：运行test_LSTM.py
训练出来的最好的模型保存在：best_model_LSTM.pth中
测试结果保存在：predictions_LSTM.txt

BERT：
训练：运行train_BERT.py
测试：运行test_BERT.py
训练出来的最好的模型保存在：best_model_BERT.pth中
测试结果保存在：predictions_BERT.txt

在运行本项目前，请对应电脑版本下载所需要的包，在requirements.txt中，直接下载可能出现不适配电脑版本的问题，可以根据requirements.txt对应自行安装，主要安装包有torch，sklearn，transformers，wrapt，PIL等。
