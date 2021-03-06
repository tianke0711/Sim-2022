##  数据部分 -- data文件夹下
### 训练集
- **train**                           原始训练集
- **train_idx**                       train基础上将label下标化
- **train_idx2**                      重新分割后的训练集
- **train_idx_stopword**              在train_idx基础上去除了停用词

### 测试集
- **test**                            原始测试集
- **test_idx**                        test基础上将label下标化
- **test_idx2**                       重新分割后的测试集
- **test_idx_stopword**               在test_idx基础上去除了停用词

### 验证集
- **val_idx2**                        没有使用交叉验证时从train_idx分离出来的验证集
- **total_idx**                       train_idx加上valid_idx数据集

### 最终测试文件
- **Sustainability_sentences_test**   最终测试文件
- **Sustainability_sentences_result** 最终测试结果文件
- **real_label**                      预测Sustainability_sentences_test.json的label
- **real_test**                       读取Sustainability_sentences_test.json的text
- **BERT_real_label**                 BERT模型预测出的标签
- **BERT_large_real_label**           BERT_large模型预测出的标签
- **ALBERT_real_label**               ALBERT模型预测出的标签
- **dBERT_real_label**                DistillBERT模型预测出的标签
- **RoBERTa_real_label**              RoBERTa模型预测出的标签
- **final_label**                     五个模型投票出的最终预测标签
- **Sustainability_sentences_final_result** 投票法的最终测试结果文件

## 文档部分 -- document文件夹下
- **config**                          存储每次运行的参数信息
- **log**                             存储每次运行的日志信息
- **model**                           存储每次运行的模型
- **preds**                           存储每次运行的预测结果

## 代码部分 -- notebook文件夹下
- **model**                           模型文件
- **prediction**                      预测文件
- **process_file**                    预处理文件
- **train**                           训练文件
