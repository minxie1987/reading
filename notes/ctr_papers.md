ctr papers
======

#### Anomaly Detection in a Large-scale Computational Advertising Platform ####

　　对于实时的RTB竞价广告系统，因为系统可能涉及到第三方系统，以及第三方系统数据的不稳定性，需要一套机制来确保当数据变化的时候，能很快的侦测出来，从而避免实时广告竞价RTB带来的不必要的损失。

　　为了监控系统的健康程度，通过定义一堆metrics来进行指标监控，甚至还需要匿名检测算法；



### Ensemble Methods for Personalized E-Commerce Search Challenge ###

total steps:

1. feature extraction:

	query-item pair ==> extract features based on search logs, browing logs, query and product metadata, other data;
	
	global statistics data:    
	a, each product show count, click count, view count, purchase count;    
	b, the total distinct user count of four types of behaviour on each product;    
	c, ctr and cvr of each product;    
	d, word length of each product;    
	e, present rank of presented product's position
	f, price, product_price_fea(A) = $$$ \frac {\\#product\\_behavior(A)} {price + 1} $$$
	
	time-based statistics features:    
	不同产品的流行度随着时间的推移会发生相应的变化；设置了周、半月、月、两个月的时间间隔；
	
	Query-item features:    
	a. category-based features:    
	b. cross token features: semantic relevance between query and product -- word level, vector level
	
	Session features:    
	a. for each query-product pair, binary session feature as whether or not the product has been clicked or viewed or purchased in the same session;
	b. further add a constant session bias to all these products in the same session when predicting(? how adding)

2. model selection:

	LR with compound feature， DNN with vector features;

3. model ensemble:

	query full scenario: LR represent relavence and DNN to represent semantic in the bottom layer, use GBDT as meta data learner;    
	![query full scenario](pics/query_full_scenario.png =450x)    
	query less scenario: do a lit change in second(bottom) layer, add a extra layer --- a model selector which is used to get the model with highest nDCG in validation set of a specific category;    
	![query less senario](pics/query_less_scenario.png =450x)

4. check results

	NDCG, final NDCG = $$$ 0.8 * NDCG_l + 0.2 * NDCG_f $$$




#### 参考文献 ####
