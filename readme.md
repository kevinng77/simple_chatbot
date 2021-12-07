# 简单旅游信息咨询系统

该系统参考了 [paddleTDS](https://github.com/cyberfish1120/PaddleTDS) ，为减少显存使用，其中的 nezha 模型使用了 3层的bert 替代，同时改进了训练方案，如改善了意图识别中正负样本比例等。

系统架构方面，加入了简易的意图询问（`feedback` 文件夹），简单的实体字典匹配（`entitylinking` 文件夹），和闲聊模块（`open_chat` 文件夹）。信息检索方面，解决了一些`paddleTSD/database_code` 中存在的bug，

## 效果截图

![image-20211207093953510](img/readme/image-20211207093953510.png)

## 快速开始

**环境配置：**

```
torch==1.7.1
transformers==4.5.1
Flask==2.0.2
Flask-Cors==3.0.10
```

**下载训练好的权重：**

链接: https://pan.baidu.com/s/1Uk11PNTCHzEdgJ7jbh5XBg  密码: kkh7
意图识别权重为 `intent_model.pth`（下载后放置于 `chatbot/intent/checkout/state_dict/intent_model.pth`）:

酒店设备识别权重为 `bi_model.pth`（下载后放置于 `chatbot/bislot/checkout/state_dict/bi_model.pth`）:

实体识别权重为 `ner_model.pth`（下载后放置于 `chatbot/ner/checkout/state_dict/ner_model.pth`）:

**启动api:**

在 `chatbot` 下执行 `python api.py`

**简易网页前端：**

浏览器打开 `web` 下的 `index.html`

## 模型训练

### 数据准备

本次数据借鉴与 [CrossWoz](https://github.com/thu-coai/CrossWOZ)，具体的数据分为：

**训练数据**

将 [CrossWOZ](https://github.com/thu-coai/CrossWOZ)/[data](https://github.com/thu-coai/CrossWOZ/tree/master/data)/**crosswoz**/ 中的 `train.json.zip`  与 `val.json.zip` 解压放置于：`CrossWOZ/data/train.json` , `CrossWOZ/data/train.json`

**信息数据库**

将  [CrossWOZ](https://github.com/thu-coai/CrossWOZ)/[data](https://github.com/thu-coai/CrossWOZ/tree/master/data)/**crosswoz**/ 中的 `database` 放置于 `chatbot\data\database`

**意图识别数据集：**

具体生成指南请参考 [chatbot/intent 文件夹]()。从CROSSWOZ提供的多轮对话数据中提取出了训练与测试集。采用CROSSWOZ对话记录的 `[General] `或 `[Request]` 对应的值作为正标签。最终数据集的正负样本比例维持在了10：1。

**实体识别数据集：**

具体生成指南请参考 [chatbot/ner 文件夹]()。从CROSSWOZ提供的多轮对话数据中提取出了训练与测试集。采用CROSSWOZ对话记录的 `[Inform] `对应的作为实体。

### 模型训练

1. 训练意图识别模型：参考 `chatbot/intent` 文件夹中 [指南](chatbot/intent/readme.md) 。

2. 添加酒店设备模型：参考 `chatbot/bislot` 文件夹 [指南](chatbot/bislot)
3. 训练实体识别模型：参考 `chatbot/ner` 文件夹 [指南](chatbot/ner/readme.md) 

## 系统结构

该对话系统参考了 [paddleTDS](https://github.com/cyberfish1120/PaddleTDS) 的框架，并做了修改与优化，增添了实体链接模块、意图反问模块。其余模块的实现方法也有多不同，如减小了预训练模型大小，提高了系统回复速度等。

<img src="img/readme/image-20211114153305267.png" alt="image-20211114153305267" style="zoom:50%;" />

系统由三大部分组成，当用户发出咨询时，信息首先在左边的query理解框架，进行领域识别，关系识别，实体识别。意图识别框架中，领域与关系同时进行分类。用匹配的方式，将多分类转换为二分类。由于数据集的特殊性，实体识别部分使用结构为3层的bert，finetune后已经可以达到98.2%的micro F1值。

中间部分为简单的信息处理，若intent识别结果不清晰，则系统将生成意图识别询问语句。

![image-20211114153554531](img/readme/image-20211114153554531.png)

当用户在询问天安们的时候，我们是很难知道用户询问的意图。因此我们可以通过实体识别的结果，当天安门被识别为景点名称后，通过一些规则，来生成意图询问语句。

实体链接方法采用的是简单的字典匹配，在对信息进行实体抽取与实体链接后，实体信息将被储存到DST字典中，作为检索条件。

最后在CROSSWOZ提供的旅游信息数据库中进行检索，生成回复语句。

## Future work

+ 添加指代消解

+ 添加依存分析

+ 参考美团KBQA领域识别的训练方案，考虑使用预精调，加入句法结构信息，蒸馏等来提高模型效果。
+ 添加额外景点信息提高系统趣味。
+ 添加观点回复功能。