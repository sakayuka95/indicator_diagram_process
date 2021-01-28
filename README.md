# indicator_diagram_process
#### 数据加载`load`
+ 读取csv或excel文件的原始二维数据
+ 从映射文件读取像素点
    + 获取映射文件数据区并生成字典
    + 根据数据区索引查找数据
+ 从示功图反推像素点
#### 异常过滤`util`
+ 数据点数不一致
+ 全零或近似零
+ 任一维为空
#### 原始示功图生成`generate_origin`
+ 从csv或excel文件的原始二维数据生成示功图
+ 从csv或excel文件的原始二维数据生成电功图
+ excel文件转换为分井数据：标准井名文件夹-行数命名txt
+ **生产参数**坐标归一化
    + 位移局部归一化（todo：入参）
    + 载荷全局归一化（todo：入参或异常过滤）
+ opencv绘制保存
+ 生成映射文件
    + 从图片
    + 从原始数据
+ 功图增强
    + 从图片
    + 从映射文件
    + ~~从原始数据~~
#### 样本生成`generate_sample`
+ 生成二分类正负样本集（训练/测试）
    + 从映射文件
    + ~~从原始数据~~
+ 生成三元组测试集
    + 从二分类测试集生成原始示功图
    + 去重
    + 生成txt测试集，格式为：a.png b.png 标签
+ 训练集中留出验证集
+ 生成二分类样本用于网络数据分类
    + 随机生成
    + 排列组合Cn2
#### 结果处理`result`
+ 训练日志转换
+ 绘制结果曲线
    + 训练验证结果csv（单项目、多项目）
    + 测试结果txt（单条、多条）
+ 二分类错误结果统计分析
+ 二分类测试样本修正
    + 从二分类测试集中删除待定部分
    + 修改待定部分样本文件名
    + 复制到二分类测试集
+ 绘制二分类测试各模型准确率
+ 查找二分类和三元组的公共错误对
#### 趋势分析`trend`
+ 绘制单口井的示功图趋势视频
+ 叠加绘制单口井示功图
#### 示功图面积计算`area`
+ 计算单张示功图像素面积
+ 计算示功图面积比率
+ 计算单张示功图真实面积
#### 通用工具`common`
+ 获取每口井的最大载荷与最大位移（csv）
+ 获取每口井的最大载荷与最大位移（excel）
+ ~~获取二分类测试集数量~~
+ 获取样本集总数
+ 获取各标签下样本数量
+ 获取txt文本行数