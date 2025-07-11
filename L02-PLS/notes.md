### partial least squares (PLS)
July 8th, 2025 15:00

- maximizing covariance between y and latent variable (linear combination of x’s), getting rid of collinearity
- works with # features > # observations
    - linear regression you cannot do this unless you regularize
    - so PLSR is somewhat a sneaky linear regression
- you can have multiple y’s
    - find subspace in X that contains information that covaries with subspace in Y
- similarly to PCA, you can do eigendecomposition on Y’X (vs. X’X)
- try not to over-interpret LVs biologically. calculate weight matrices AND regression coefficients, so there is an extra degree of freedom
    - LVs capture phenotypically relevant variance
- some algorithms will deflate (subtract previous transformations) X and Y as the latent variables are being found, others won’t
    - so solutions not necessarily unique
- cross validation → determine # of LVs looking at Q2Y goodness of prediction
    - for very vigorous, do train test split, then CV on the train split
- differential expression analysis
    - instead of using highly variable genes filter, can use PLSR and then perform subsequent downstream analysis on the LVs (weights for each gene) themselves
    - but be careful: what is the difference between statistical significance and biological significance?
        - even if you use logFC, what threshold to use? and why would we use the same one for every gene?
    - don’t even think about building an autoencoder for > 10k genes if u don’t have around 800 samples
- “biologically-driven dimensionality reduction” example → instead of training model on gene counts, infer transcription factor activity (essentially linear combinations of groups of genes) and then train model on that
- some people use LASSO for feature selection, then PLSR on the smaller subset of features. but be careful – if LASSO finds two features equally important, it’ll toss out one at random. LASSO could be good if you are doing biomarker discovery + interested in generalizable model, whereas PLSR is maybe more interpretable for if you are trying to understand a specific dataset and how the features correlate with phenotype (though you technically can extrapolate with it too)
