一道排序引发的思考
=====

> 整理Learning to Rank模型的一些基本思想和工程处理方式
>
> ps. 这里不是基本算法里面的数组排序问题哦，嘿嘿！


　　像美团、点评这种生活类，都有很多商铺或者物品，等待用户去搜索、查看，选择最适合自己的东西，对于公司运营来说，这里需要一个合理的排序方式去排序，并展现给用户；假定现在每个商铺只有评分信息，请问如何根据评分信息去进行排序，或者说这种排序方式有什么问题，请问你怎么去改进？

　　这个是在面试的时候碰到一个开放性问题，当时其实没有回答得特别好；不过也顺便发现了些东西，就是如果让你去设计一个系统，或者回答一个这种类型的开放性问题，第一个要做的不是去想，而是先找到笔和纸[除非有很强很强的逻辑和记忆能力，不需要笔和纸就能够列出所有情况]，然后在纸上逐个列出条件和问题，以及逐个解决，先把每个单独的问题解决出来了，然后将他们汇总结合，形成一个合理的统一整体。

#### non-model 思路 ####

　　我们先给出一种非模型的思路，然后对这种思路进行优缺点解析；非模型思路，其实也就是人工来定规则，比如认为数量重要，那么我们简单的数量乘以一个权重，然后将所有计算的得到的一个值，然后通过归一化处理，映射到一个范围[比如0 - 5的范围]就可以的了。

　　在真实世界中，可不会这么简单了，比如如果我们需要考虑评分的时间[基于这样一个假设，最近的一个评分要比很早以前的评分重要得多]，另外我们还需要考虑评论的数量，由于这两个是在不同维度上表现，并且不同的维度表现意义是不同的，那么也就不是简单的一个权重就能完事了的，需要在不同维度设计不同的公式，并且希望最终能够归一化的一个数值范围内；比如我们可以在时间上通过时间衰减的方式，确定每个评分的得分，然后进行归一化操作；而对于数量的评分的话，我们可以通过对数的方式，将数值控制到一定范围；比如我们可以使用下面这个公式:
$$ score\_{item} = log\_{10}\\{num\\_people\\} + \frac{{\sum}\_{i=0}^{N}\frac{score\_{i}}{(t+2)^{gravity}}}{N} $$，
通过这种综合得分的方式，得到每个物品的得分，然后直接排序即可；如果有其他需要添加考虑的因素，通过这种方式将得分计算，并最后在必要的时候归一化操作即可。[这里只是给出了一种简单的量化方式，上面这种方式也不是没有问题，比如不同用户不同的评分可能影响是不一样的]

[ps. 如果对时间衰减感兴趣的话，可以参照hacker news或者reddit的post排序算法，[中文介绍hacker news和reddit的排序算法](http://www.cnblogs.com/zhengyun_ustc/archive/2010/12/15/amir.html) ]

　　上面这种通过各种因子添加的方式，如果因素比较少，且每个因素都有比较明显的物理意义的话，从实现的角度以及可解释的角度来讲，会比模型或者其他隐含的方式直观得多，同时不需要过多的计算量，同时也综合考虑了各种因素，排序也相对中规中矩[均衡]；这种方式的缺点是如果我们要考虑更多的因子、比如个性化，比如展示的时候也会根据搜索的关键词来匹配，这时候如果仍然使用这种方式，就会显得有点力不从心，甚至是没办法对某一些因子进行比较好的物理意义上"衰减"和"归一化"。

#### model 思路 ####

　　模型的思路，是指用learn to rank的方式[排序学习]，学习一个比较好的模型，也就是我们所说的，用机器学习的方式来考虑很多因素的排序，而不是逐个通过人工设置参数的方式添加进去，让机器去自动寻找合适的因素和参数；

　　排序是信息检索中的核心问题，同时在其他地方也会涉及到排序问题，比如广告展示、个性化推荐，也都会或多或少的跟排序有关；信息检索中排序的方式主要有以下两类：

1. 重要性排序，根据文档本身的重要程度来排序，而不用管与查询词的关系；
2. 相关度排序，选择这样一种度量，也就是查询词和文档之间的相关性，来对文档进行排序，比如布尔模型、余弦相似度等；

　　我们上面提到的非模型思路，其实就是重要性排序，也就是根据商铺本身评分的重要性，然后进行排序；这两种方法的缺点是随着考虑的因素越来越多，进行人工进行选择的方式会变得越来越不现实。而通过机器学习的方式，会把上面两种排序方式的所有因素当做特征，然后将这些放在一起当做训练样本，由机器学习的方式，自动学习特征的重要性，以便能够将多个因素都考虑进来。

　　排序学习模型，也是一种监督性学习方式，这就要求我们在模型训练之前先要对数据进行准备，特征转换，以及选定合适的评估函数进行模型参数的训练。

##### 数据准备 ####

数据准备包括数据获取、数据的标注、特征获取以及数据的清洗。

###### 数据获取 ######

　　获取训练训练数据，以便后续的数据处理和模型的训练；数据获取的方式有两种，一种是人工标注，另外一种是从日志中获取；

1. 人工标注，即通过人工的方式标注训练样本的排序顺序，或者在根据查询词查询好的结果中，选择文档进行相关性标注，这种方式的缺点是依赖于人的判断标准，并且不同的人，判断标准可能不一样，尤其是在比较模糊的一些概念上；
2. 日志中挖掘，日志不会像人工标注那样，有明确的标注样本，日志中一般记录的都是用户的各种操作行为，我们可以通过对这些行为进行清理和转换，然后在一定条件下使得这些数据能够充当数据标注的作用；具体的操作方法请参照Joachims$$$ ^{[2]} $$$。

###### 数据标注 ######

 　　同其他监督学习方式类似，我们需要在训练之前对数据进行标注，标注训练样本数据的类别；对于排序问题，我们可以分为以下三种类型的标注：
 
 1. PointWise：点的方式，也就是直接标注每个文档的绝对相关值，比如我们可以对相关性用类别{Perfect, Excellent, Good, Fair, Bad}，有了类别之后，我们训练就可以使用多分类模型训练方式来进行训练，比如最大熵、SVM；
 2. PairWise：Pair，也就是成对的方式，将排序问题转换为二元分类的问题，$$$ \\{(x\_1, x\_2, +1), (x\_3, x\_4, -1)\\} $$$，这种模型的训练方法有Rank SVM，RankNet等；
 3. ListWise：上面两种都是讲排序问题转换为分类或者回归问题，而是直接选择一个合适的度量函数，直接对文档的排序结果进行训练和评估优化；
 
 [ps. 后面我们再来说明下每个方式的优缺点]

###### 特征获取 ######

　　有了数据之后，我们需要筛选出可以使用的特征，以便ranking model训练的时候进行使用；在文档查询的使用中，我们既要关注文档、查询词本身的特征，同时也要给出其他一些和当前排序模型相关的特征，比如查询词好当前文档的一些交叉信息-相关度信息等，前面提到的非模型思路的每个度量都可以作为特征加入到模型训练中来。

###### 数据清洗 ######

　　数据清洗时为了避免噪音数据给模型训练带来一些误导，从而使得模型在训练数据上看着准确，但是在未知数据上进行预测的时候，却表现非常糟糕。可以使用的数据清洗方法有:

1. 通过对数据每个特征的值，进行值的统计和查看，查看值是否有异常值、缺失值；
2. 通过统计分布，查看数据统计分布是否在基本逻辑上是否合理[比如95%的值属于同一值，那么这个特征就没必要加入到最终模型训练中]；
3. 必要时通过聚类方法，查看离群点，对离群点进行查看和校验，查看是否有必要清洗掉；
4. 结合本身的业务理解，数据是否可用；
5. 数据的归一化、标准化处理；

##### 模型评估与训练 #####

###### 模型评估 ######

　　对Ranking Model的评估，和其他模型训练算法类似，通过比较实际的输出结果和模型的预测结果，以及量化他们之间的差异，通过这种方式来评估模型的好坏。比较常见的Ranking Model的评估方法有，NDCG、DCG、MAP以及Kendall's Tau。

　　首先我们定义下各个表示符号：

1. $$$ q\_i $$$：当次查询的查询词；
2. $$$ D\_i $$$：查询词$$$ q\_i $$$查询得到的查询文档结果；
3. $$$ \pi\_i $$$：查询文档结果集合$$$ D\_i $$$的排列组合，即$$$ \pi\_{i}(j) $$$表示的文档j在查询结果中对应排列位置；
4. $$$ y\_i $$$：表示每个单独文档和查询词$$$ q\_i $$$的相关度，可以使分类的度量[perfect, good, normal, bad...]，也可以是得分[2.3]；

DCG：
$$ DCG(k) = \sum\_{j:\pi\_{i}{\leq}k}G(j)D({\pi}\_{i}(j)) $$，
其中$$$ G\_{i}({\cdot}) $$$表示受益函数，表示和当前查询的一个相关重要程度；$$$ D\_{i}({\cdot}) $$$表示位置折扣函数，表示对位置的一个惩罚，越到前面惩罚越小，越到后面惩罚越大；$$$ {\pi}\_{i}(j) $$$表示文档$$$ d\_{i,j} $$$在文档排列$$$ {\pi}\_i $$$中的位置；DCG度量的是前$$$ k $$$个位置的累积求和；

NDCG，在DCG基础上进行归一化操作：
$$ N{\,}DCG(k) = G\_{max, i}^{-1}(k) \sum\_{j:\pi\_{i}{\leq}k}G(j)D({\pi}\_{i}(j)) $$，
NDCG主要是为了解决在DCG的基础上不同查询词的数目，返回的关联文档数目不一致的问题；

　　Gain函数，可以简单定义一个指数函数，目的是希望相关性越高的类别，对应的Gain得分也越高，Gain函数的定义如下：
$$ G(j) = 2^{y\_{i,j}} - 1 $$，
其中$$$ y\_{i,j} $$$表示的是对应文档$$$ d\_{i,j} $$$的相关度度量得分。

　　位置折扣函数(Position discout function)，定义这个折扣函数的目的是希望这个折扣函数，能够跟位置有个负相关的关系，即随着位置的增加，给出分数应该是逐渐变低，最简单的方式是$$$ \frac{1}{pos+1} $$$，这里使用了对数函数，同样是单调递增函数，添加上对数是希望下降影响不要那么大，而是逐渐平缓且到一个点之后，位置影响会变得不那么重要[就像在搜索引擎中，很少有人在看到最后一页的最后几条]，对应的函数如下：
$$ D({\pi}\_{i}(j)) = \frac{1}{log_{2}(1+{\pi}\_{i}(j))} $$，
其中$$$ {\pi}\_{i}(j) $$$表示的是文档$$$ d\_{i,j} $$$在当前检索排序中的位置；

那么对应，完整的DCG和NDCG公式为：
$$ DCG(k) = \sum\_{j:\pi\_i(j){\leq}k}\frac{2^{y\_{i,j}} - 1}{log\_2(1+{\pi}\_{i}(j))} $$
$$ N\,DCG(k) = \_{max, i}^{-1}(k) \sum\_{j:\pi\_i(j){\leq}k}\frac{2^{y\_{i,j}} - 1}{log\_2(1+{\pi}\_{i}(j))} $$

MAP

　　MAP(mean average precision)，是在IR中使用较多的一种度量方法；说到MAP，我们得先说AP[MAP不就是在AP的基础上加个均值M处理嘛]，AP(Average Precision)的定义如下，在模型的评价指标中，我们经常看到ROC、AUC，在连续的值上，AP其实是和AUC等价的，因为AUC表示的是precision-recall曲线，也就是在recall上求积分就是Precision的累积值，因为recall的范围是[0, 1]，类似于添加了1的归一化操作，也就是如下公式:
$$ AP = {\int}\_{0}^{1}p(r)dr $$；
在ranking model上，由于每个文档是一个个离散的组成的，对应的AP计算如下：
$$ AP = \frac{{\sum}\_{j=1}^{n\_i}P(j){\cdot}y\_{i,j}}{{\sum}\_{j=1}^{n\_i}y\_{i,j}} $$，
$$$ y\_{i,j} $$$表示$$$ d\_{i,j} $$$与查询词相关与否[0, 1]，$$$ P(j) $$$ 定义为：
$$ P(j) = \frac{{\sum}\_{k:{\pi}\_i(k){\leq}{\pi}\_i(j)}y\_{i,k}}{{\pi}\_i(j)} $$，
用$$$ P(j) $$$表示小于文档$$$ d\_{i,j} $$$位置$$$ j $$$的文档相关性的准确度，其中对应的每个相关与否只用0、1表示。


RC

　　 RC(Rank Correlatio)，常见于Kendall‘s Tau。

###### 模型训练 ######

　　我们使用监督学习方法对Ranking Model进行模型训练，通过定义损失函数，并在损失函数上进行优化操作，使得损失函数达到一个"最小值"。Ranking Model的损失函数和一般监督学习模型的损失函数不一样，因为Ranking Model更加关注的是查询的结果是否达到一个"合理"的顺序，而不是达到一个合理的分数值；而根据对排序损失的不同定义，有以下三种主要方式:

1. PointWise，只需要关注每个文档和查询词的相似度得分即可，定义损失函数为：$$$ L(F(x), y) = {\sum}\_{i=1}^{n}(f(x\_i) - y\_i)^2 $$$，一般的回归或者分类方法就可以处理这种模型训练，比如Logistics Regression, SVM；
2. PairWise，点对方式的损失函数可以使用hinge loss、指数损失函数、logistics regression，可以在rankSVM、RankBoost、RankNet、IRSVM中使用，对应的损失函数为：$$$ L(F(x), y) = {\sum}\_{i=1}^{n-1}{\sum}\_{j=i+1}^{n}{\phi}(sign(y\_i - y\_j), f(x\_i) - f(x\_j)) $$$；
3. ListWise，不同的方法，对应的损失函数不太一样，比如AdaRank的损失函数为：$$$ L(F(x), y) = exp(-NDCG)] $$$；其他方法有SVM MAP、LambdaRank、SoftRank、GPRank、CCA、RankCosine、ListNet、ListMLE，以及LambdaMART；

这三种方法的优缺点:

1. 

##### [参考文档] #####

[1] A Short Introduction to Learning to Rank. Li Hang. IEICE,2010    
[2] Optimizing Search Engines using Clickthrough Data. Thorsten Joachims. SIGKDD,2002.    
[3]  Learning to Rank for Information Retrieval. Tie-yan Liu.    