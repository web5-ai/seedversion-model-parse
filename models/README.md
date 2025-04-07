# 模型实现 

记录一下当前模型相关的要点

**目前问题可能出在图像处理上，应该是个好解决的问题，看客户啥时候有空再聊一下我们**

1. 所有模型使用严格模式加载都可以通过并进行预测，目前的模型结构应该是没问题的，主要问题是预测结果偏差非常大
2. 图像预处理，包括缩放、归一化。暂时不清楚客户在输入张量前是怎么进行图像预处理的，但是图像预处理切实会影响模型输出
3. 输出顺序，不清楚输出时的数据格式是 [蛋白质, 油脂] 还是反过来，根据用户给的训练代码看，顺序是[油脂, 蛋白质]
4. 模型输出时的归一化，模型输出前，最后有一个归一化的过程，在调试过程中我标注出每个归一化的代码

- Swin.py 125 使用sigmoid进行归一化，归一化后还会乘100直接放大成百分比量

  ```
  def func_transition(self,outputs):

      # outputs = outputs.view(-1)

      outputs[:,0]= (torch.sigmoid(outputs[:,0])*100)

      outputs[:,1]= (torch.sigmoid(outputs[:,1])*100)

      return outputs[:, :2]
  ```
- ResNet18.py 21 这个是我自己复现的结构，但是能完全适应状态字典。只是用的线性层把输出格式转换成2了

  ```
  class ResNet18(nn.Module):
      def __init__(self, num_classes=2):  # 默认输出类别数为2，适用于本地模型
          super(ResNet18, self).__init__()
          # 加载预训练的ResNet18模型
          self.ResNet = resnet18(pretrained=False)  # 这里设置为False，因为我们将加载本地的预训练模型
          # 增加一层全连接层，用于将ResNet18的输出转换为数量输出
          self.fc1 = nn.Linear(self.ResNet.fc.out_features, num_classes)

      def forward(self, x):
          # 前向传播，先通过ResNet18提取特征，然后通过全连接层得到数量输出
          x = self.ResNet(x)  # 提取特征
          x = self.fc1(x)  # 转换为数量输出
          return x
  ```
- VanillaNet.py 24 和Swin是一样的方法
- MPViT.py 30 和Swin一样
- EfficientNet: model_zoo.py 94 和Swin一样
- FasterNet: model_zoo.py 418 和Swin一样

## 测试工作安排
