深度学习更新方法
======


> 各种神经网络更新方法、优缺点说明


#### SGD ####

随机梯度下降法

优点：

实现简单?

缺点：

所有方向上的梯度更新都是一样的比例，也就是学习率在各个方向上一样的，没有考虑到快接近最小值的时候，需要调整步长，使得很容易在最小值的"沟壑"周围来回跳动；

以下情况，也会容易出现问题：

areas where the surface curves much more steeply in one dimension than in another；

#### Momentum ####

动量惯性更新方法

$$$ v_t = \gamma v\_{t-1} + \eta\triangledown\_{\theta}J(\theta) $$$    
$$$ \theta = \theta - v\_t $$$

It does this by adding a fraction γ of the update vector of the past time step to the current update vector；

惯性的意思就是利用前一次的动量信息；

helps accelerate SGD in the relevant direction and dampens oscillations；

$$$ \gamma $$$通常指设置为0.9，不是特别正式的说法，把$$$ \gamma $$$称作为动量；

```
# Momentum update
v = mu * v - learning_rate * dx # integrate velocity
x += v # integrate position
```

As an unfortunate misnomer, this variable is in optimization referred to as momentum (its typical value is about 0.9), but its physical meaning is more consistent with the coefficient of friction. 

Effectively, this variable damps the velocity and reduces the kinetic energy of the system, or otherwise the particle would never come to a stop at the bottom of a hill. 

When cross-validated, this parameter is usually set to values such as [0.5, 0.9, 0.95, 0.99]. Similar to annealing schedules for learning rates (discussed later, below), optimization can sometimes benefit a little from momentum schedules, where the momentum is increased in later stages of learning. A typical setting is to start with momentum of about 0.5 and anneal it to 0.99 or so over multiple epochs.

优点

为了解决SGD扰动的问题;

通过这种惯性的设置，使得在坡度陡的地方能较大的步长，坡度平缓的地方使用较小的步长，避免来回扰动，通过这种方式来加快收敛；

缺点

![momentum updater](pics/dl_momentum_update.png =550x)

#### Nesterov ####

如果把梯度下降比喻为一个球向下滚动的话，momentum则是在下降过程中不断进行动量的累积；随机梯度则是沿着固定的加速度向下滚动；

$$$ v\_t = \gamma v \_ {t - 1} + \eta \triangledown \_ {\theta} J(\theta - \gamma v\_{t - 1}) $$$    
$$$ \theta = \theta - v\_t $$$

和momentum方法相对，Nesterov方法进行超前计算，由于我们知道当前动量和下一次动量的位置，那为什么不可以直接超前计算到下一个更新点的梯度呢，而不是使用前一次的梯度信息；

```
x_ahead = x + mu * v
# evaluate dx_ahead (the gradient at x_ahead instead of at x)
v = mu * v - learning_rate * dx_ahead
x += v
```

或者从上一次来看，那对应的表述方式为

```
v_prev = v # back this up
v = mu * v - learning_rate * dx # velocity update stays the same
x += -mu * v_prev + (1 + mu) * v # position update changes form
```

#### 自适应方法(adaptive methods) ####

#### Adagrad ####

```
# Assume the gradient dx and parameter vector x
cache += dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)
```

Notice that the weights that receive high gradients will have their effective learning rate reduced, while weights that receive small or infrequent updates will have their effective learning rate increased. 

#### Adadelta ####

extension of adagrad that seeks to reduce ites agressive, monotonically decreasing learning rate. 

Adadelta restricts the window of accumulated past gradients to some fixed size w.

$$$ \Delta \theta_t = - \dfrac {RMS[\Delta \theta]\_{t-1}}{RMS[g]\_{t}} g\_{t} $$$

$$$ \theta\_{t+1} = \theta\_t + \Delta \theta\_t $$$

其中 RMS 为:    
$$$ RMS[\Delta \theta]\_{t} = \sqrt{E[\Delta \theta^2]\_t + \epsilon} $$$

E为:    
$$$ E[\Delta \theta^2]\_t = \gamma E[\Delta \theta^2]\_{t-1} + (1 - \gamma) \Delta \theta^2\_t $$$

With Adadelta, we do not even need to set a default learning rate, as it has been eliminated from the update rule.

#### RMSprop ####

对adagrad的调整:

```
cache = decay_rate * cache + (1 - decay_rate) * dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)
```

Here, decay_rate is a hyperparameter and typical values are [0.9, 0.99, 0.999].


#### Adam ####

```
m = beta1*m + (1-beta1)*dx
v = beta2*v + (1-beta2)*(dx**2)
x += - learning_rate * m / (np.sqrt(v) + eps)
```

Recommended values in the paper are eps = 1e-8, beta1 = 0.9, beta2 = 0.999. 

推荐为一种默认的方法，或者使用组合SGD+Nesterov Momentum


#### Annealing the learning rate ####

- step decay, Typical values might be reducing the learning rate by a half every 5 epochs, or by 0.1 every 20 epochs。
- Exponential decay, like this form $$$ \alpha = \alpha \_ 0 e^{-kt}  $$$
- 1/t decay, like this form: $$$ \alpha = \frac {\alpha \_ 0} {1 + kt} $$$

#### 参数优化 ####

神经网络的训练，最重要的几个参数有：

1. 初始的学习率
2. 学习率的衰减方式
3. 正则化的强度，L2惩罚，dropout的强度；

选择一个验证集进行数据的验证，而不需要使用多次交叉验证；    
超参数的选择：

- learning rate:

	```
	learning_rate = 10 ** uniform(-6, 1)
	```
	
	raising it to the power of 10;

- regularization strength，正则化强度

	和learning rate的方式一样；

- dropout = uniform(0,1), 前面的learning rate 和 regularization strength是在一定范围内进行参数的选择和调优，dropout则通过原始的方式操作；
- 选择random search，而不是grid search;
- Careful with best values on border, check whether is the edge of this interval or not, or miss more optimal hyperoarameter setting beyond interval.
- Stage your search from coarse to fine, 从粗到细力度的方式进行参数的搜索，coarse range(e.g. 10 ** [-6, 1]), coarse search while only training for 1 epoch or even less. fine stage -- narrower search with 5 epochs.
- Bayesian Hyperparameter Optimization,  Spearmint, SMAC, and Hyperopt, still relatively difficult to beat random search in a carefully-chosen intervals$$$ ^{[3} $$$.


其他方面的经验：

- Same model, different initializations
- Top models discovered during cross-validation
- Different checkpoints of a single model
- Running average of parameters during trainings

	to maintain a second copy of the network’s weights in memory that maintains an exponentially decaying sum of previous weights during training



#### 调优经验 ####


对应的优化方法的"下山"演示

![down hill](pics/dnn_opt2.gif =400x)

![down hill](pics/dnn_opt1.gif =400x)

http://ycszen.github.io/2016/08/24/SGD%EF%BC%8CAdagrad%EF%BC%8CAdadelta%EF%BC%8CAdam%E7%AD%89%E4%BC%98%E5%8C%96%E6%96%B9%E6%B3%95%E6%80%BB%E7%BB%93%E5%92%8C%E6%AF%94%E8%BE%83/

If your input data is sparse, then you likely achieve the best results using one of the adaptive learning-rate methods. An additional benefit is that you won't need to tune the learning rate but likely achieve the best results with the default value.

RMSprop is an extension of Adagrad that deals with its radically diminishing learning rates. It is identical to Adadelta, except that Adadelta uses the RMS of parameter updates in the numinator update rule. Adam, finally, adds bias-correction and momentum to RMSprop. Insofar, RMSprop, Adadelta, and Adam are very similar algorithms that do well in similar circumstances.     
Adam might be the best overall choice.

if you care about fast convergence and train a deep or complex neural network, you should choose one of the adaptive learning rate methods.

#### Parallelizing and distributing SGD ####

#### references ####

[1]. http://cs231n.github.io/neural-networks-3/    
[2]. Random Search for Hyper-Parameter Optimization    
[3]. http://nlpers.blogspot.com/2014/10/hyperparameter-search-bayesian.html    
[4]. ADADELTA: An Adaptive Learning Rate Method. Retrieved from http://arxiv.org/abs/1212.5701    
[5]. http://sebastianruder.com/optimizing-gradient-descent/index.html    
