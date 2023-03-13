# 我的面经 · Python

​	本人面经之 **Python** 分栏，后续补充。

​	主要是 Python 相关技巧经验。



# 基础

## 技巧 

- \r 本行开头（可用 print( , end='\r') 不断从头打印）\n 下一行开头

- 格式化字符串 f'' 的用法 https://blog.csdn.net/yizhuanlu9607/article/details/89530982

- `max(l, key=lamda x: abs(x))`：可直接获取绝对值最大的 x

- `exec(string)` 运行字符串 

- 连接推荐 `str.join(list)`，更快

- 字符串链接不用加什么 `'py''thon' => 'python'`

- 字典默认值 `d.get('a', []) `如果有 'a' 则返回value，否则返回 []

- `%%time`打印整个单元格的壁时间，而`%time`只提供第一行的时间。使用`%%time`或`%time`打印2个值：①CPU时间；②壁厚时间

- 海象运算符`:=`（3.8+特性）https://blog.csdn.net/qq_40244755/article/details/102685199

- 转换编码

  ```python
  def re_encode(path):
      with open(path, 'r', encoding='GB2312', errors='ignore') as file:
          lines = file.readlines()
      with open(path, 'w', encoding='utf-8') as file:
          file.write(''.join(lines))
  ```

- `sum` 用于合并列表：https://blog.csdn.net/weixin_39564036/article/details/111286230

- 不要忘了 `while i in range(n)` 的用法

- print 中`,` 和 `+` 不一样，前者会得到 a b，后者则是 ab

- split 默认空格

- 可用 `*` 解包，`' '.join()` 也可以获得同样效果

  ```python
  lists = [[1, 2, 3], [4, 5, 6]]
  for i in lists:
      print(*i)
      
  # Out:
  # 1 2 3
  # 4 5 6
  ```

- 顺时针旋转矩阵 90°：`zip(*matrix[::-1])`；逆时针旋转矩阵 90°：`list(zip(*matrix))[::-1]`。

- `zip` 源码：https://docs.python.org/zh-cn/3/library/functions.html#zip

- `send`传值生成器：https://www.jianshu.com/p/6c33bd958f3d

- 批量生成变量 https://mp.weixin.qq.com/s/wTRNlu_naggmvumAqRb6PA

- more

## 模块

- 程序包里放 `__init__.py` 即使是空
  - ①初始化模块
  - ②将所在目录当 `package` 处理
  - ③`from import` 导入子包可以用 `__all__` 加相关模块名字字符列表，导入指定文件，而避免导入系统相关文件
  
- 当前目录优先级高于 `sys.path`

- `from sortedcontainers import SortedList, SortedSet, SortedDict` 有序数据结构

- `collections.defaultdict()` 和 `dict.setdefault`的使用 https://blog.csdn.net/yangsong95/article/details/82319675

- Bisect https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/submissions/

- 在Python中也可以用语言特性 @lru_cache 来自动缓存。 

  ```python
  from functools import lru_cache
  
  @lru_cache(None)
  ```

- `time.strptime` 将时间字符串变成，时间相关的元组 `time.strftime` 获取元组，得到时间

- 动态加载模块 https://www.jb51.net/article/57694.htm

## 对象

- **类变量**，用类调用，用于人类寿命此类与具体实例无关的变量
- @classmethod 构建 **类方法**，类直接调用，用途与类变量相似，即不因对象而改变的则定义为类xx，否则为实例xx
- @staticmethod 构建 **静态方法**，无法获取类、实例变量，故只能做逻辑独立的小功能，例如加工字符串
- 推荐定义为半私有，方便单元测试
- 多继承时，当多个父类有同名方法或者变量时，子类继承左边的父类
- 不推荐多继承，这会增加代码的复杂度

## 垃圾回收（GC，Garbage Collection）

计数引用，遇上循环引用故有 —— 标记清除、分代回收

https://www.jb51.net/article/52229.htm

https://www.cnblogs.com/panwenbin-logs/p/13531119.html

http://c.biancheng.net/view/5540.html

# 装饰器

- [@property 与 @staticmethod 装饰器的介绍与使用](https://blog.csdn.net/weixin_41888257/article/details/107563335)





# 正则

- http://regex101.com/

- 正则速查表



# Jupyter

- Jupyter lab
- Jupyter notebook



# NumPy

- 深度之眼 Python_NumPy.ipynb；工具库_NumPy.ipynb
- shape、切片 取到的是视图
- 整数、bool 取到的是副本
- shuffle 打乱原数组，permutation 不打乱原数组，返回新数组
- 所谓 axis 是沿着轴方向



# SciPy

- 工具库_SciPy.ipynb
- SciPy 的 线性代数库linalg 比 NumPy 的全面
- 关于稀疏矩阵
  - dok_matrix（字典保存） 和 lil_matrix（双列表保存） 适合逐渐添加元素
  - coo_matrix 使用等长的三个数组保存元素（可重复），但不支持增删查改，若操作只能转为其他矩阵



# Matplotlib

- 深度之眼 Python_Matplotlib.ipynb；工具库_Matplotlib.ipynb
- API 基本同 Matlab（所以觉得 Matplotlib 文档烂可以去看 Matlab中国）
- 可渲染对象（简单类型和容器类型）基类 Artlist抽象类
  - 简单类型是标准的绘图组件，如：Line2D、Rectangle、Text、AxesImage等
  - 容器类型可以包含多个组件，主要将这些组件组织起来，如：Axis、Axes、Figure等
- 建议按照对象顺序绘图（例如 Figure -> Axes -> axis -> label）
- 复用有官方的 Gallery
- 有高级封装 `import seaborn as sns`
- `subplot(323)` 和 `subplot(3,2,3)` 是相同的



# Pandas

- 深度之眼 Python_Pandas.ipynb；工具库_Pandas.ipynb
- Series（存储单列数据）和 DataFrame（存储多列数据）是最常用的两个对象
  - Series 具有数组和字典的能力
  - DataFrame 每一列中格式相同，列和列可以不同
- 取值 loc、iloc、at、iat、lookup，其中 lookup 可以用两个列表以此得到对角元素
- 有些判断可以用 query 代替
- read_csv 时 encoding 的 utf-8 不等于 utf-8-sig（开头带了 BOM，用于标识此文件为 utf-8）
- 分组运算：根据特定条件，将数据分为多个组，类似正则中的 group
- **其实dataframe本身就是一种降维操作，把 n 个特征展示在二维表格上，类似于三维投影在二维，他们把多个维的数据投影在一个固定维上，获得了一堆数据。是一种 reshape。**



# Scikit-learn

- 深度之眼 Python_Scikit-learn.ipynb；工具库_Scikit-learn.ipynb

- 标准化数据

  ```python
  # 从sklearn.preprocessing导入StandardScaler  
  from sklearn.preprocessing import StandardScaler  
  # 标准化数据，保证每个维度的特征数据方差为1，均值为0，使得预测结果不会被某些维度过大的特征值而主导  
  ss = StandardScaler()  
  # fit_transform()先拟合数据，再标准化  
  X_train = ss.fit_transform(X_train)  
  # transform()数据标准化  
  X_test = ss.transform(X_test) 
  ```

- 

- 



# TensorFlow 2

- 建议在 Colab 学 TF，本家，你懂的

- 30 天吃掉 TensorFlow https://github.com/lyhue1991/eat_tensorflow2_in_30_days

- 手撸 AI 算法

- 莫烦 PYTHON https://mofanpy.com/

- 吴恩达 TensorFlow 视频

  视频：https://www.bilibili.com/video/BV1qE411u7z4?p=40

  练习：https://github.com/lmoroney/dlaicourse

- 《动手学深度学习》 https://github.com/Shusentang/Dive-into-DL-PyTorch

- GPU计算以张量（Tensor）为单位

- more



# Pytorch

- 20天吃掉那只 Pytorch https://github.com/lyhue1991/eat_pytorch_in_20_days
- numpy 数组和 torch tensor 的区别 https://blog.csdn.net/qq_28368377/article/details/103096377
  1. torch 的 tensor，有逗号还有 tensor 字母
  2. tensor 的 empty 生成的数偏小一点，而且 numpy 的 empty 要用 列表参数
  3. tensor 默认 int64、tensor.float32，numpy 默认 int32、float64
  4. tensor 的 size 方法是返回一个 size 方法对象；numpy  的是 int
  5. 可以互转：
     - 视图：<tensor>.numpy() 是 tensor 转 array；torch.from_numpy(<array>) 是 array 转 tensor
     - 副本：torch.tensor(<array>) 是 array 转 tensor



# Colab

- 基本指令

  ```python
  ! /opt/bin/nvidia-smi
  ---
  ! git clone ......   .git
  ---
  import os
  os.chdir(“/content/...”)
  ---
  %run ....  .py
  ---
  from tensorflow.python.client import device_lib
  device_lib.list_local_devices()
  ---
  %tensorflow_version 2.x
  ---
  wget http://
  ```

- `tf.distribute.cluster_resolver.TPUClusterResolver` 使用 TPU  

  ​		https://blog.csdn.net/big91987/article/details/87898100

  - 设置 TPU

  - 初始 TPU 环境

  - 转化为 TPU 模型

    ```python
    try:
        tpu = tf.contrib.cluster_resolver.TPUClusterResolver() # TPU detection
        strategy = tf.contrib.tpu.TPUDistributionStrategy(tpu)
        train_model = tf.contrib.tpu.keras_to_tpu_model(model, strategy)
    except ValueError:
        tpu = None
        train_model = model
        print('Running on GPU or CPU')
    ```

- more

- 



# PyQt5 开发

1. 安装 Pycharm
2. 创建项目
3. 项目中安装 pyqt5、pyqt5-tools
4. 外部工具设置 Qt Designer、pyuic（参数：\$FileName\$ -o \$FileNameWithoutExtension$.py）



# 问题

- E:\资料\计算机\软件\NLP\极客时间NLP训练营\机器学习面试100题\Python面试基础.pdf

- E:\资料\计算机\软件\算法\Python面试 128

