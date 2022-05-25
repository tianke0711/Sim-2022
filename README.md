##  数据部分
- train 原始训练集
- test 原始测试集
- train_idx train基础上将label下标化
- test_idx test基础上将label下标化
- train_idx2 重新分割后的训练集
- test_idx2 重新分割后的测试集
- total_idx train_idx加上valid_idx数据集
- train_idx_stopword 在train_idx基础上去除了停用词
- test_idx_stopword 在test_idx基础上去除了停用词
- val_idx2 没有使用交叉验证时从train_idx分离出来的验证集
- real_label 预测Sustainability_sentences_test.json的label
- real_test 读取Sustainability_sentences_test.json的text

## 文档部分
