## 酒店设备识别

参考了 [paddleTDS](https://github.com/cyberfish1120/PaddleTDS) 对酒店设备限制进行检测。

### 生成训练数据集

`cd bi_utils && python data_processer.py`

### 训练

在 `bislot` 文件夹下执行 

```
python main.py --epoch=10
```

### 检验结果

随机挑选验证集中的案例预测：

案例："好的，北京万达文华酒店有吹风机吗？"

```
python predict.py
```

预期结果：

> ['酒店-酒店设施-吹风机']