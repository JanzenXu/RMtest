# RMtest

| 许骏哲 | 人工智能学院 | 2021 | 智能科学与技术 |
| :----: | :----------: | :--: | :------------: |

## Primary

- 初级题目目标检测基本完成
- 目标跟踪打算使用Camshift算法完成，但是一直搞不出来，==不知道怎么目标检测和目标跟踪怎么嵌套==
- 初级题目全部用的是OpenCV，没有用到神经网络
- 我自己测试代码使用的是群里数据集图片合成的视频，目标检测的准确度还可以，==但场外的光源还是会有影响==，这个后期可以通过获取摄像头角度解决
- 使用提供的视频测试时可以识别到装甲板，但是场地灯光对算法的影响很大，经常出现误识别的情况，如果在实机上操作可以通过修改镜头参数、获取摄像头角度解决解决一部分问题
- 增加了调试图像二值化的阈值的test程序，可以根据实际情况调整合适的阈值

## Intermediate

- 中级题目基本完成
- 预留了窗口移动函数，后期可以根据能量开关旋转速度、子弹下坠等计算出提前量
- 主要问题就是五个开关全被点亮后，能量开关闪动时窗口会一直跟踪，但实际操作中全部点亮后应该可以手动退出，问题应该不大

## Practice

- 由于时间问题，练习题只做了一部分，只完成了红线检测
- 十字路口检测可以用角点检测，在一个矩形区域内有4个角点即可认为该矩形区域包含一个十字路口
- 数字识别我用过OpenCV里ml模块的KNN算法，但是效果不好就没用





