## 意图识别

参考了 [paddleTDS](https://github.com/cyberfish1120/PaddleTDS) ，对用户意图进行类别检测。使用匹配的方式，将多分类方式转换成了二分类。

### 生成训练数据集

执行 `chatbot/intent/intent_utils` 下的 `data_processer.py`

```
cd intent_utils && python data_processer.py
```

### 训练

回到 `cahtbot/intent` 文件夹下执行 

```
python main.py --epoch=2
```

每个epoch耗时约20分钟（在3070笔记本上）

### 检验结果

随机挑选验证集中的案例预测：

案例："你好"

```
python predict.py
```

预期结果：

> ['greet-none']