数学基础知识
=======

> 机器学习中各种数学基础概念

#### 协方差矩阵 ####

　　covariance matrix， 协方差矩阵的每个元素是各个向量之间的协方差；其i, j位置的元素是第i个与第j个随机向量之间的协方差，是标量随机变量到高维度随机向量的自然推广。

$$$
\sum = E[(X-E[x])(X-E[X])^T]
$$$


　　协方差矩阵的性质:

- $$$ \sum = E(XX^T) - uu^T $$$
- $$$ \sum $$$是半正定、对称矩阵
- $$$ \sum \geq 0 $$$
- $$$ \sum = \sum^T $$$
- 若$$$ X $$$与$$$ Y $$$独立，则有$$$ cov(X, Y) = 0 $$$



{ps. 在mou中latex公式的使用，行内公式是使用三个$, block是使用两个$, 但是在hexo中行内和block则是不同的，行内使用一个$, block是使用的两个$}


#### 共轭梯度法 ####

　　用于求解凸优化问题，常见的求解方法是通过梯度下降法，固定步长挥着通过搜索步长的方式，进行逐步迭代求解；还有一种就是通过牛顿法和拟牛顿法，这些方法不止利用了梯度的方向，还在梯度的方向选择最优的长度进行搜索，而这个最优长度是选择的二阶来选择，而不是像梯度下降法那样选择一个固定步长，或者通过探索的方式选择最优步长。

　　共轭梯度法，则是另外一种不同的求解最优化问题的方法；是介于最速下降法[梯度下降] 与 牛顿法之间的一个方法，它仅需利用一阶导数信息，但克服了最速下降法收敛慢的缺点，又避免了牛顿法需要存储和计算Hesse矩阵并求逆的缺点。

　　首先解释下什么是共轭方向：

　　设A是n x n对称正定矩阵，若$$$ \Re^n $$$中的两个方向$$$ d^{(1)} $$$和$$$ d^{(2)} $$$满足$$ d^{(1)T}Ad^{(2)} = 0 ， $$则称这两个方向关于A共轭，或称它们关于A正交。[正定矩阵，对于任何非零向量z，都有$$$ z^TMz > 0 $$$]

　　在上述关于共轭的定义中，如果A是单位矩阵，则两个方向关于A的共轭等价于两个方向正交[共轭是正交概念上的推广]；

　　共轭梯度法最初由Hesteness 和 Stiefel 在1952年为求解线性方程组而提出的；后面，人们把这种求解方法应用于无约束最优化问题，从而变成了一种重要的最优化算法；共轭梯度法的基本思想，是把共轭性与最速下降方法结合，利用已知点处的梯度构造一组共轭方向，并沿这组方向搜索，求出目标的极小点。$$$ ^{[1]}　$$$

我们先讲解一个简单的二次凸函数的共轭梯度法，然后将这个方法推广到极小化一般函数，考虑以下问题，

$$ min f(x) = \frac{1}{2}x^TAx + b^Tx + c $$，其中$$$ x \in \Re^n $$$，A是正定矩阵，c是常数。

对于梯度下降法是，我们沿着梯度的方向进行搜索，通过这种方式逐步收敛到极小值附近，从而通过这种方式近似的求解最优化问题；具体的搜索方式如下：
　　$$ x^{(k+1)} = x^{(k)} + \lambda\_{k}d^\{(k)} $$，
其中步长$$$ \lambda\_{k} $$$满足
$$ f(x^{(k)} + \lambda\_{k}d^{(k)}) = minf(x^{(k)}\ |\ {\lambda}d^{(k)}) $$，
也即
$$ {\varphi}({\lambda})  = f(x^{(k)} + {\lambda}d^{(k)}) $$，
求$$$ {\varphi}({\lambda}) $$$的极小点，也就是一阶导数为0，
$$ {\varphi}^{'}({\lambda}) = {\nabla}f(x^{(k+1)})^Td^{(k)} = 0 $$，
将上面的内容结合原始函数得到如下内容：
\begin{eqnarray}
(Ax^{(k+1)} + b)^Td^{(k)} &=& 0 \nonumber \newline
[A(x^{(k)} + {\lambda}\_{k}d^{(k)}) + b]^Td^{(k)} &=& 0 \newline
[g_k + {\lambda}\_kAd^{(k)}]^Td^{(k)} &=& 0 \\qquad(1)
\end{eqnarray}

[ps. 公式中最后一项是从倒数第二项展开得到的，$$$ Ax^{(k)} + b = g\_{k}$$$ ]    
从公式(1)得到：
$$ {\lambda}_k = -\frac{g\_{k}^Td^{(k)}}{d^{(k)T}Ad^{(k)}} $$
若通过迭代的方式没有收敛，则使用$$$ -g\_{k+1} $$$和$$$ d^{(k)} $$$来构造下一个搜索方向，并使得$$$ d^{(k+1)} $$$和$$$ d^{(k)} $$$关于矩阵A共轭，令
$$ d^{(k+1)} = -g\_{k+1} + \beta\_{k}d^{(k)} $$
上式两端左乘$$$ d^{(k)T}A $$$，
$$ d^{(k)T}Ad^{(k+1)} = -d^{(k)T}Ag\_{k+1} + {\beta}\_{k}d^{(k)T}Ad^{(k)} = 0 $$，
并求得$$$ {\beta}\_{k} $$$，
$$ {\beta}\_{k} = \frac{d^{(k)T}Ag\_{k+1}}{d^{(k)T}Ad^{(k)}} $$，
通过这种方式进行不断的求得正确的梯度方向，然后不断迭代求解即可，这种方法在二次凸函数上具有二次终止性，也就是两次搜索能够达到极小点，这里不证明，有兴趣可以参照参考文献[1]。

求解方式，我们先选择一个初始点，然后得到梯度方向，以及求得步长$$$ {\lambda}\_{k} $$$，或者通过$$$ {\beta}\_{k} $$$确定下一次梯度的方向，然后通过梯度信息确定下一次迭代的步长。对于正定的二次函数，$$$ {\beta}\_{k} $$$有如下性质：
$$ {\beta}\_{k} = \frac{{\parallel}g\_{(i+1)}{\parallel}^2}{{\parallel}g\_{(i)}{\parallel}^2} $$

　　共轭梯度法的推广，将共轭梯度法推广到一般极小化任意函数，与上面的二次凸函数的共轭梯度不同的是关于步长$$$ {\lambda}\_{k} $$$的确定，这里步长的确定是通过一维搜索的方式进行求得，另外一个是矩阵A的地方，是使用现行点出的二阶Hessian矩阵替代。

　　FR的共轭梯度法的计算步骤如下：

![nn_hessain_algo](pics/common_cg.png =550x)    
![nn_hessain_algo](pics/common_cg_2.png =550x)

对应的$$$ \beta \_ {j} $$$的求解方式，有一下几种：    
![nn_hessain_algo](pics/cg_betas.png =250x)

#### Jacobian ####

　　Jacobian matrix(雅可比矩阵)，是函数的一阶偏导数排列的矩阵，其行列式成为雅可比行列式：

\begin{eqnarray}
J = \frac {df}{dx} = [\frac {\partial f}{\partial x\_1} \, \dots \, \frac {\partial f}{\partial x\_n}] = 
\left ( \begin{array}{ccc}
\frac {\partial f\_1} {\partial x\_1} & \ldots & \frac {\partial f\_1} {\partial x\_n} \newline
\vdots & \ddots & \vdots \newline
\frac {\partial \large{f}\_m} {\partial \large{x}\_1} & \ldots & \frac {\partial \large{f}\_m} {\partial \large{x}\_n} \newline
\end{array} \right)
\end{eqnarray}


#### 矩阵求导 ####

#### 各种损失函数 ####

　　损失函数，是用来度量目标值和预测值之间的差异的；不同的损失函数对应的意义是不一样的；

1. hinge loss：$$$ \ell(y) = max(0, 1 - t \cdot y) $$$，用在SVM最大分类间隔中；
2. epsilon insensitive loss：$$$ \ell(y) = max(0,|w'x - y| - epsilon) $$$，在svm regression中有相应的使用；$$$ ^{[2]} $$$


#### 坐标下降法 ####

　　coordinate descent，非梯度优化方法，算法在每次迭代中，在当前点沿一个坐标方向进行一维搜索以求得一个局部最小值；

#### Hybrid Monte_Carlo Sampling ####

　　和一般的Monte_Carlo方法不同的是，

#### positive-definite ####

　　正定矩阵，假设M为n阶方阵，如果对于任意非零向量z，都有$$$ z^TMz > 0 $$$，那么对应的矩阵M就是正定矩阵；正定矩阵具有的性质有：

1. 矩阵M的所有特征值都未正的；
2. 每个正定矩阵都是可逆的；
3. rM也是正定矩阵，$$$ M + N $$$，$$$ MNM $$$，$$$ NMN $$$都是正定矩阵，如果矩阵M，N都是正定矩阵；
4. ...

#### 线性代数基础知识 ####

##### 基本符号 #####

使用的符号有：

1. $$$ A \in R^{m \times n} $$$，表示一个$$$ m \times n $$$的矩阵(m行n列)。
2. $$$ X \in R^n $$$，含有n个元素的向量，n行1列的矩阵，即列向量，如果表示一个行向量，则使用$$$ X^T $$$来表示。
3. $$$ a\_{ij} $$$表示第i行 第j列的元素，$$$ a\_{j} $$$表示第j列，$$$ a\_{i}^T $$$表示第i行。

##### 矩阵、向量的乘法 #####

1. $$$ C = AB \in R^{m \times p} $$$，其中$$$ A \in R^{m \times n}，B \in R^{n \times p} $$$。
2. 点积

	$$ x^Ty = [ \begin{matrix} x\_1 & x\_2 & ... & x\_n \end{matrix} ] \left[ \begin{matrix} y\_1 \\\\ y\_2 \\\\ ... \\\\ y\_n \end{matrix} \right] = \sum\_{i=1}^{n}x\_{i}y\_{j} $$

3. 外积，$$$ xy^T $$$
4. 矩阵和向量的乘积，
5. 矩阵与矩阵的乘积，可以看做是向量-向量的乘积，矩阵-向量的乘积，向量-矩阵的乘积

##### 运算和性质 #####

1. 单位矩阵和对角矩阵
2. 转置，$$$ (A^T)\_{ij} = A\_{ji} $$$
3. 对称矩阵，$$$ A = A^T $$$，反对称矩阵$$$ A = -A^T $$$
4. 矩阵的迹(trace)，$$$ trA = \sum \_{i=1} ^{n} A\_{ii} $$$，矩阵的trace的性质：

	![矩阵的迹](pics/matrix_trace.png =500x)

5. 范数
6. 线性无关和秩
7. 矩阵的逆，$$$ AA^{-1} = I = A^{-1}A $$$，非方阵没有逆，方阵不一定有逆；

	满足下面条件是可逆的：
	- $$$ (A^{-1})^{-1} = A $$$
	- $$$ (AB)^{-1} = B^{-1} A^{-1} $$$
	- $$$ (A^{-1}) ^{T} = (A^{T}) ^{-1} $$$

8. 正交矩阵，$$$ x^T y = 0 $$$，$$$ U^T U = I = U U^T $$$
9. 矩阵的值域和零空间，
10. 行列式
11. 二次型和半正定矩阵，
12. 特征值和特征向量，$$$ Ax = \lambda x $$$，$$$ x $$$是特征向量，$$$ \lambda $$$是特征值；矩阵的迹为特征值之和，矩阵的行列式为特征值之积；对称矩阵的特征值和特征向量，
13. 矩阵微积分，

	- 梯度，
	![matrix gradient](pics/matrix_gradient.png =350x)
	- hessian矩阵，
	![matrix hessian](pics/matrix_hessian.png =350x)
	- 最小二乘法，$$$ Ax = b $$$，解为$$$ x = (A^T A)^{-1} A^T b $$$


##### 凸优化 #####

1. 定义，$$$ x, y \in C，0 \le \theta \le 1 $$$，有$$$ \theta x + (1 - \theta)y \in C $$$，文字描述是两个点在C中，那么他们的连线也是在C中，
2. 特征，convex set的交集仍然是convex set，半正定矩阵也是convex set；
3. 凸函数：$$$ f(\theta x + (1 - \theta)y) \le \theta f(x) + (1 - \theta)f(y) $$$

	$$$ f(y) \ge f(x) + \triangledown \_{x} f(x)^T (y - x) $$$    
	![function convex](pics/function_convex.png =350x)    
	$$$ \triangledown \_{x} ^2 f(x) \succeq 0 $$$    
	jensen不等式：    
	$$$ \alpha $$$-sublevel:    
	
4. 凸优化问题：

	minimize f(x)    
	subject to $$$ x \in C $$$
	
	或者改写为
	
	minimize f(x)    
	subject to $$$ g\_{i}(x) \le 0, i = 1,...,m $$$    
	           $$$ h\_{i}(x) = 0, i = 1,...,n $$$

5. 



#### 参考文献 ####
[1] 最优化理论和算法    
[2] http://kernelsvm.tripod.com/
