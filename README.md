# Classification-using-KMM-Kernel-Mean-Matching-
Dataset Bias correction (Python)

Please note: Complete dataset is not uploaded due to file size restriction (<=25mb) by github but small subset of the complete dataset is still available for testing and training.

In this project, I used Kernel Mean Matching (KMM) algorithm for bias-correction in a dataset and find out the accuracy of classification algorithm.

What is Bias?
Many real-world applications exhibit scenarios where training and test data are drawn from the same distribution. But there might be some scenarios where training and test data are drawn from different distributions. This is referred to as Sample Selection Bias, that is, training data distribution are biased compared to test data distribution.

What is KMM?
Kernel Mean Matching (KMM) is a well-known method for bias correction by estimating density ratio between training and test data distribution. This mechanism re- weights training data instances so that their weighted data distribution resembles that of the observed test data distribution.

Dataset courtesy of Computer Science Department, The University of Texas at Dallas:
Only one smallest size instance of a training dataset and one smallest size instance of a testing dataset is uploaded.


Useful paper about KMM:
 J Huang, A Gretton, KM Borgwardt, B Schölkopf, AJ Smola.
Correcting sample selection bias by unlabeled data. In Advances in neural information processing systems, pages 601-608, 2006.
