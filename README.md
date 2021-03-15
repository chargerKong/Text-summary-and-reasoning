# Text-summary-and-reasoning
汽车大师文本摘要与推理

运行方式
建立data文件夹, data下建立stopwords和 word2vec文件夹, 把哈工大停用词表放入stopwords中,

下载AutoMaster_TestSet.csv 与 AutoMaster_TrainSet.csv 到data文件下下.
https://aistudio.baidu.com/aistudio/datasetDetail/1407

运行utils下的data_loads文件,生成所需文件,再 训练

## seq2seq的baseline
首先是基准seq2seq+attention的模型结构，原文的Pointer-Generator Networks也是在此基础上构建的，框架如图1所示：
![image](https://user-images.githubusercontent.com/49099366/111097698-fce16180-857c-11eb-89b0-077a21acd059.png)

正常来讲，Seq2Seq的模型结构就是两部分--编码器和解码器。正常思路是先用编码器将原文本编码成一个中间层的隐藏状态，然后用解码器+注意力机制来将该隐藏状态解码成为另一个文本。

再解码过程中，会出现OOV问题（out of vocabulary），为解决此问题

## Pointer-Generator Networks
原文中的Pointer-Generator Networks是一个混合了 Baseline seq2seq和PointerNetwork的网络，它具有Baseline seq2seq的生成能力和PointerNetwork的Copy能力。如何权衡一个词应该是生成的还是复制的？原文中引入了一个权重 $p_{gen}$

![image](https://user-images.githubusercontent.com/49099366/111098245-03bca400-857e-11eb-8ef7-a25c7f660c35.png)

但是结果还有词语重复出现的问题，下面加入
##  Coverage Mechanism
运用了Coverage Mechanism来解决重复生成文本的问题，下图反映了前两个模型与添加了Coverage Mechanism生成摘要的结果：
![image](https://user-images.githubusercontent.com/49099366/111098356-41213180-857e-11eb-9251-21ca2821b97c.png)
蓝色的字体表示的是参考摘要，三个模型的生成摘要的结果差别挺大。红色字体表明了不准确的摘要细节生成(UNK未登录词，无法解决OOV问题)，绿色的字体表明了模型生成了重复文本。为了解决此问题--Repitition，原文使用了在机器翻译中解决“过翻译”和“漏翻译”的机制--Coverage Mechanism(具体参考《Modeling Coverage for Neural Machine Translation》)。
