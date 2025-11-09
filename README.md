# Classification-models

This is an experimantal report on implementing and analyzing fundamental machine learning classification models. It is divided into two main sections.

### The first section

Focuses on weighted logistic regression, beginning with the mathematical derivation of the gradient vector and Hessian matrix for the weighted logistic loss function. It introduces a complete implementation of a simple logistic regression model in Python, followed by an extended version that integrates sample-specific weights based on a Gaussian kernel parameterized by τ. The section further explores the influence of varying τ values (0.01–5) on model behavior, showing how smaller τ values make the model more sensitive to local data variations and potential noise, while larger τ values promote smoother, more global decision boundaries.

### The second section

The second section compares Naive Bayes and Support Vector Machine (SVM) classifiers on the MNIST handwritten digits dataset. The Naive Bayes model is implemented with Laplace smoothing and Gaussian probability estimation, achieving approximately 81% accuracy. The SVM part examines multiple kernel functions (Linear, RBF, and Polynomial) under One-vs-All and One-vs-One multi-class strategies, analyzing the effect of different C regularization values. Empirical results show that RBF and Polynomial kernels improve accuracy as C increases, with the best-performing configuration (RBF kernel, C=1) achieving 94% accuracy. Notably, both multi-class strategies yield nearly identical results.
