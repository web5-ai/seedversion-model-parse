# fastapi后台

创建时间2025.3.1
创建人：QG-rise

## 后台功能设计

- model_api.py:
  封装ModelApi类，接口服务启动时（目前设计是单线程的）初始化，内部维持一个model_loader变量，提供eval_image()接口
  eval_image(self, image, model_name='ResNet'|'VGG'|'FasterNet', model_path=None)->dict
  该接口接收图片image，模型名称model_name，模型路径model_path，其中image是待预测的图片（该变量可能是url、文件路径以及图片文件，最终会在方法内转换为图片文件BytesIO类型），model_name用于选择使用哪种结构以加载状态字典，model_path选择使用哪个状态字典（一般默认通过config.py配置该参数）
- main.py:
  async def predict(image_url: str, model_name='ResNet'|'VGG'|'FasterNet', model_path=None):
- 注：test_model.py基本没参考价值了，以及整个项目里有很多重复代码，在使用时我就以model_loader.py为主要参考了，尽量以这个文件里实现的方法为基准

## 功能测试记录

现在先测试几个模型能否正常调用

- ResNet: 有效
- Swin: bug还未调试好
- FasterNet: 有效
- EfficientNet:
- MPViT: 有效
- VanillanNet:  调整后，已经有效
