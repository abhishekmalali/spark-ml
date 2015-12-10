# Ordinal Regression
## Motivation
In order to model categorical data there are two possible approaches:
- multinomial logistic regression
- ordinal regression
Either of these models are an extension of the famous binary logistic regression model to allow multiple labels instead of two only.

#### Multinomial logistic regression
In Multinomial logistic regression the dependent variable $Y$ is modelled taking values in category $c = c_1, ..., c_C$  via

$$ \mathbb{P}(Y_i = c_j) = \frac{\exp(\beta_K X_i)}{1 + \sum_{k=1}^K \exp(\beta_k X_i)}$$

Both sklearn and Mlib implement this approach to some extent. I.e. sklearn can handle multinomial logistic regression, but uses a lbfgs or newton-cg approach only (no sgd) with support for L2 regularization solely. In Mlib 

However, multinomial logistic regression is not always the best model to choose. Consider i.e. the case of ratings. Here, the different categories represent ordinal values implying some kind of natural order. In a multinomial logistic regression model this order is neglected. I.e. rating '5' is as good as '4' and just another category.

Links:
- <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>
- <http://spark.apache.org/docs/latest/mllib-linear-methods.html>
- <http://de.slideshare.net/dbtsai/2014-0620-mlor-36132297>

### Why Ordinal regression?
To make up for this missing order, another model that is not yet implemented in sklearn nor Mlib is ordinal regression. The following is based mainly on Chapter 9 (Tutz, G. (2012). Regression for categorical data. Cambridge: Cambridge University Press.) 

#### Types of ordinal regression
Suppose we are given a response captured by a random variable $Y \in \mathbb{R}$ which is categorial with w.l.o.g. (a mapping always exists) $K$ different categories $\lbrace 1, ..., R \rbrace $ with their natural order and furthermore
$$ \mathbb{P} ( Y \in \lbrace 1, ..., R \rbrace) = 1$$

We model the ordinal regression model using binary variables $y_1, ..., y_R$ (i.e. $y_r \sim \mathrm{Ber}(p_r) \quad \quad 0 \leq p_r \leq 1$) which will be used to transform an ordinal response variable into a binary response. This way, ordinal regression can be seen as an extension to a classical, binary variable model.

For the transformation of the ordinal response to a binary one, there are three different, commonly used approaches(let $r =1, ..., R$):

- Cumulative model:
    $$ y_r = \begin{cases}1 \quad \quad Y \in \lbrace 1, ..., r \rbrace \\ 0 \quad \quad Y \in \lbrace r+1, ..., K\rbrace\end{cases}$$
- Sequential model:
    $$ y_r\vert_{\lbrace Y \geq r \rbrace} = \begin{cases}1 \quad \quad Y =r \,\vert\, Y \geq r \\ 0 \quad \quad Y >r \,\vert\, Y \geq r \end{cases}$$
- Adjacent model:
    $$ y_r\vert_{\lbrace Y \in \lbrace r, r+1\rbrace \rbrace} = \begin{cases}1 \quad \quad Y = r \,\vert\, Y \in \lbrace r, r+1\rbrace \\ 0 \quad \quad Y = r+1 \,\vert\, Y \in \lbrace r, r+1\rbrace \end{cases}$$ 
The most versatile model of these is the cumulative assuming nothing in particular about the data except the ordinal regression assumptions. The sequential model might be used to model variables for which higher values are more likely to be reached if the lower ones have already occured. A similar argumentation holds for the adjacent model with neighbors.

### Cumulative model (Threshold model)
The cumulative model (or threshold model) can be deviated assuming a latent, underlying r.v. $\tilde{Y}$ for which the actual, observed variable $Y$ can be seen as somehow rounded.

The link between those variables shall be modelled through
$$ \lbrace Y = r \rbrace = \lbrace \theta_{r-1} < \tilde{Y} \leq \theta_r \rbrace$$
for $-\infty = \beta_0 < \beta_1 < ... < \beta_{R-1} < \beta_R = +\infty$
For the latent variable a linear regression model<font color='red'>add here footnote: Sometimes this also modelled as $$ \tilde{Y} = x^Tw + \epsilon $$ leading to different signs in the cdf. However, semantically both approaches are equivalent.</font> with $x, w \in \mathbb{R^n}$ and a random variable $\epsilon$ is choosen:

$$ \tilde{Y} = - x^Tw + \epsilon $$
It follows that
$$ \mathbb{P}(y_r = 1 \vert x, w) = \mathbb{P}(Y \leq r \vert x, w) = \mathbb{P}(-x^Tw + \epsilon \leq \beta_r) = \mathbb{P}(\epsilon \leq \beta_r + x^Tw) = F_\epsilon(\beta_r + x^Tw)$$

$$ \implies \mathbb{P}(Y = r) = F_\epsilon(\beta_r + x^Tw) - F_\epsilon(\beta_{r-1} + x^Tw)$$
using the cdf $F_\epsilon$ of $\epsilon$.

In the model setup the only thing left to define is what distribution to assume/choose for $\epsilon$. Some possible choices which yield different popular models are:

- cumulative logit model: $$ \epsilon \sim \mathrm{Logistic}(0,1)$$
- probit model: $$ \epsilon \sim \mathcal{N} (0, 1)$$
- Maximum extreme-value model: $$ \epsilon \sim \mathrm{Gumbel}(0, 1)$$
- Minimum extreme-value model: $$ -\epsilon \sim \mathrm{Gumbel}(0, 1)$$

![probability density functions of different error distributions](img/epsdf.png)

Whereas the probit and logit model are a good choice for symmetric errors, the Gumbel and reversed Gumbel distribution can be used for the case of asymmetric error distributions to provide a better fit. For many cases, a certain assymetry regarding the error is quite natural.

### Visualizing the intuition
![Categorical data can not be fitted via a linear regression model](img/orddata.png)
Assume we are given data that is distributed according to the plot above. As we see the data is somehow categorical distributed. Fitting a linear model (red) via OLS does not really make sense. However, thinking of a distribution of individual categories that are somehow linked together seems to be a reasonable assumption.
![](img/val4.png)
I.e. looking at the empirical distribution of all $X$ for the category $Y = 4$ suggests that either a reverted Gumbel or probit model might be a good idea.

