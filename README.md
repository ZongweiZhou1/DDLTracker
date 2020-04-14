
```
Distractor-Aware Discrimination Learning Model for Online Multiple Object Tracking
```

# 方法框架
![DDL 框架图](figures/ddl%20framework.png)
主要针对于SOT处理MOT任务中容易发生漂移问题.其中$L_{cls}$ 表示SOT跟踪中模板与搜索区域的匹配度, $L_{reg}$表示目标回归损失, $L_{det}$表示检测属于行人的概率.


$$
L_{reg} =L_{br}(t, t^*) + \beta L_{cs}(b)
$$
其中$L_{br}$和FasterRCNN中定义相同, L_{cs} 表示预测的box的方差.

$$
L_{cls} = L_{cls}^e(p, p^*) + \alpha w_iL_{cls}^h(p_i, p_i^*)
$$
其中 $L_{cls}^e$表示和FasterRCNN中损失相同, $L_{cls}^h$表示hard negative samples的损失. hard negative samples定义为与目标的IOU小于0.5且与其他的目标IOU大于0.5. 

$$
L_{det} = L_{det}(d, d^*)
$$
表示每一个achor对应到行人的损失. 这里的正样本是与任意一个目标的IOU大于0.5, 负样本与所有的目标的IOU都小于0.4. 

下图给出了1d场景下,我们处理MOT任务相对于SOT中anchor正负样本选择的不同示意图.
![anchor selection](figures/anchor%20selection.png)

下图给出了Relation attention module (RAM)的示意图.

![relationship attention module](figures/relation%20attention%20module.png)


# 代码使用
```bash
sh make.sh
python eval_mot.py
```


# 声明
跟踪部分代码来源于 MOTDT 项目.



