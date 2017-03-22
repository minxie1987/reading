模型、实验、检验
========

### experiment at airbnb ###

　　介绍了airbnb在使用实验对产品进行改进的时候，遇到的一些坑(pitfall)，以及他们的一些解决方案。首先说明的是实验的目的是为了验证产品的想法、改进是否可行；以及在快速实验的过程中，airbnb所遇到的一堆坑。

对应的"坑"如下：

1. 一个实验需要跑多久才算有效？

	根据不同的sample size来决定实验的时间，并且事先得有一个大概的概念；对于避免二类错误(存伪-false negative，一类错误为弃真-false positive)，最好的方式是确定最小有效数量，也就是计算sample size，具体的sample size计算的建议，可以参照[参考\[2\]](http://www.evanmiller.org/ab-testing/sample-size.html)。但是对于有问题的实验，则是越早停掉越好，先排除bug的问题，然后再进行实验，排除其他因素的影响；

	airbnb是结合时间和p-value来确定一个实验是否需要停掉[不同的时间 不同的p-value]；

	> 　　We wrote code to simulate our ecosystem with various parameters and used this to run many simulations with varying values for parameters like the real effect size, variance and different levels of certainty. This gives us an indication of how likely it is to see false positives or false negatives, and also how far off the estimated effect size is in case of a true positive. 

2. 对于实验的结果，需要结合context来进行理解，不一定是好的实验就会有好的表现，需要结合实际context对是实验结果好坏的分析；将实验结果分成几组有意思的小组，然后深入查看各个小组的实验效果；
3. 确保实验平台本身的正确性，可以使用A/A的方式来分析本身分流、实验等的正确性；

关于实验的那些坑，可以观看这个[视频](https://www.youtube.com/watch?v=lVTIcf6IhY4).


关于实验的文章，其他可以参照的文章

Online Controlled Experiments at Large Scale    
Trustworthy Online Controlled Experiments: Five Puzzling Outcomes Explained    
Seven Pitfalls to Avoid when Running Controlled Experiments on the Web    

### How Not To Run An A/B Test ###

　　内容来自[链接](http://www.evanmiller.org/how-not-to-run-an-ab-test.html)。

### reference ###

[1] http://nerds.airbnb.com/experiments-at-airbnb/    
[2] http://www.evanmiller.org/ab-testing/sample-size.html    