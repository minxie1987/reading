RNN/LSTM
=====

#### LSTM ####

　　对于传统的RNN，可以通过循环的方式，记住一定量的上下文context信息，比如“云在天空中”；这是RNN的优点，而不是一般的神经网络一样，只有空间上的信息。

　　但是，由于RNN本身循环的方式是使用很简单的方式，而这种简单的方式使得他不能够记录较长的信息(long dependency)，

![simle rnn](pics/simple_rnn.png =450x)

当gap比较大的时候，比如图中的$$$ x\_0 $$$ 和 输出 $$$ h\_{t+1} $$$，一旦距离变大之后，对于RNN能够记录到这种信息，就会变得复杂、困难很多。

　　而LSTM则是通过gating strategy的方式使得能够记住和必要时候的信息"遗忘"，通过这种"有的放矢"针对性的学习，使得不需要那么多参数或者复杂度而达到这种long-term dependency的效果。也就是通过在链路连接的时候通过一些相对复杂的gate来实现：

![](pics/simle_lstm.png =450x)

LSTM的核心思想:

1. Cell, 传送纽带，将t - 1时刻 和 t时刻连接起来

	![lstm cell](pics/lstm_cell.png =350x)
	
	通过sigmod函数来控制是否需要遗忘前一次的状态，如果遗忘前一次状态，那么当前cell的输出信息则只包含当前信息。
	
	而在整个一次的chain上，则有三个地方来控制cell的状态；

2. cell的三次状态控制：

	![cell controller 1](pics/lstm_cell_1.png =450x)
	
	通过当前状态和上一次状态结合进行sigmod转换，通过0-1来控制是否需要遗忘上一次cell state.
	
	![cell controller 2](pics/lstm_cell_2.png =450x)
	
	通过sigmod的输入门(input gate layer)决定信息是否需要进行更新累加到cell state上去；通过tanh非线性函数来创建一个候选更新值；
	
	![cell controller 3](pics/lstm_cell_3.png =450x)
	
	将old cell state和当前的候选更新值进行累加，得到当前状态下的cell state.
	
	![cell controller 4](pics/lstm_cell_4.png =450x)
	
	根据当前状态的sigmod激活函数和当前状态下的cell state, 决定当前的输出值$$$ h\_t $$$.

	![full chain](pics/lstm_full_chain.png =450x)
	
	完整的一个lstm的chain block.
	
	其他LSTM的变种, 常见的两种变种如下:
	
	![lstm variant 1](pics/lsmt_variant_1.png =450x)
	
	![lstm variant 2](pics/lstm_variant_2.png =450x)


#### 网络训练 ####

  传统的cnn，甚至是深度的CNN神经网络，使用的前向、后向传播算法进行参数计算的；通过定义损失函数、以及残差传播、梯度下降更新的方式进行参数的更新。

#### stacked lstm ####
