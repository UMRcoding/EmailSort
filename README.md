# 开源项目说明

**目的：学术研究**

如果您遇到任何安装或使用问题，可以通过QQ或issue的方式告诉我。同时使用本项目写论文或做其它学术研究的朋友，如果想把自己的研究成果展示在下面，也可以通过QQ或issue的方式告诉我。看到的小伙伴记得点点右上角的Star~

![](https://umrcoding.oss-cn-shanghai.aliyuncs.com/Obsidian/202212032313952.png)





# 内容过滤

## 一、基于内容过滤

1. 通过在抓取每个商品的一系列特征来构建商品档案

2. 通过用户购买的商品特征来构建基于内容的用户档案

3. 通过特定的相似度方程计算用户档案和商品档案的相似度

4. 推荐相似度最高的n个商品。所以，这种推荐基于与已购买商品的相似度来进行推荐


    最初，这种系统用于文档推荐如网络新闻, 网页以及书籍。 用户档案和商品档案都以使用信息提取技术或信息过滤技术提取出的关键词集合来表示。鉴于两个档案都以权重向量的形式表示，则相似度分数则可以使用如余弦近似度方程或皮尔森相关系数等启发式方程来计算得到。其它的技术如分类模型，构建一个统计方法或者数据挖掘方法，来判断文档内容和用户是否相关。

基于内容过滤局限：

1. 不容易找到足够数量的特征来构建档案（特征缺少问题）
2. 推荐内容局限于目标用户已购买商品（超特化问题）

3. 还未有购买记录的新用户或偏好特殊的用户不能得到合适的推荐(新用户、特殊用户问题) 



## 二、协同过滤

1. 从每个用户对商品的评级信息中构建用户档案

2. 使用如余弦相似度、皮尔森相关系数或距离函数来识别和目标用户**具有相似意向的用户**，他们对商品有相似的评级

3. 对来自具有相似意向用户的偏好评级取均值、加权和或调整后的加权和，推荐n个商品


这种方法基于用户之间的相似性来进行推荐。这种评级预测的方法称为基于记忆的方法。其它的评级预测方法为基于模型的方法，这种方法从大量的评级数据上建立概率模型和机器学习模型来预测商品的评级。基于协同过滤的推荐系统目前有很多优化改进，包括推荐新闻的Tapestry算法，网络新闻的GroupLens算法，针对音乐的Ringo算法。

协同过滤推荐的局限如下：

1. 对于还未给商品评级的用户无法进行商品推进（新用户问题）

2. 对未被评过的商品进行推荐也有难度（新商品问题）

3. 评级信息缺乏时推荐效果较差（稀疏问题）

   

## 三、基于规则的方法

    还有计算简单且流行的推荐方法为基于规则的方法。使用数据挖掘技术从大量的过往交易数据中获取规则。它可以是会同时被购买的商品之间的关联规则，也可以是按时间依次被购买商品的序列模型。基于规则的推荐方法的主要局限为难以为没有在关联规则或序列模型中出现的商品进行推荐。Aggarwal提出了一种针对目标市场的发现局部关联规则的技术。他们首先聚类分析了来自UCI机器学习数据中的蘑菇数据集和成人数据集两个购物篮数据，然后从每个类别中提取关联规则。Huang提出了一个序列模式推荐系统来预测超市中顾客随时间变换的购买行为。



## 四、混合方法

        混合推荐系统目的在于减少乃至克服基于内容推荐、协同过滤和基于规则的推荐系统的局限。Fab系统联合了协同过滤和基于内容过滤技术来消除基于内容过滤技术中的特征缺乏和超特化问题以及协同过滤中的新商品问题。在这个系统中，基于内容的用户档案依旧用来寻找相似的用户来进行协同推荐，商品会在以下两个条件同时满足时推荐给用户：（1）被推荐商品在目标用户档案中有较高的分数；（2）被推荐商品在目标用户的相似用户中有较高的评级。Liu对购物篮数据使用二变量选择分析（购买／未购买）聚类并选出k个近邻，从k个近邻中的购买频次来获得商品（未被目标用户购买）得分的预测值。同时，根据新的隐式用户评级信息来从整个用户空间来选择近邻，并根据这些近邻的评级的调整加权和来给出商品（未被购买或已被购买）得分的预测值。另外，他们将整个时间划分为三段，并对每个时间段的交易数据进行聚类分析，然后得到由三个阶段顺序交易数据聚类得到的序列模式，由此得到整个时间段由一系列商品代表的序列模式。因此，这种方法比其它方法更优在于可以做更过的个性化推荐。



向量空间模型

神经网络模型

评论观众垂直度

评论数据指数增长的关系

对直播画面和语言的迅速检测快速反应

知乎 瓦力保镖评论保护



## 实验思路

1. 读取数据集：使用`email.parser.BytesParser`解析`EmailMessage`对象，以查看邮件文本内容，并批量获取邮件内容。
2. 数据预处理：构造函数获取邮件的结构类型及其计数，确定邮件包含的类型，对训练集分词，获得邮件分词结果，用英文停用表进行过滤，nltk词干提取
3. 特征提取：将文本，文档集合转换为矩阵，从测试集中计算词汇和IDF，得到词语，词频，并计数特征向量。
4. 使用学习的词汇和文档频率，得到训练集特征向量
5. 计算每封邮件是垃圾邮件的贝叶斯概率
6. 预测结果正确率















