# Coefficient of Determination (R²)

The Coefficient of Determination, commonly denoted as R², is a statistical measure used to assess the goodness of fit
of a regression model. It represents the proportion of the variance in the dependent variable that is predictable from
the independent variables.

R² provides a scale from 0 to 1, where higher values indicate a better fit and imply that the model can better explain
the variation of the output with the input variables. It is particularly useful for comparing the explanatory power of
different models.

## Implementation Details

R² is calculated as the proportion of the total variation of outcomes explained by the model. Mathematically, it can
be represented as:

$$
R² = 1 - \frac{\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{N}(y_i - \bar{y})^2}
$$

where $y_i$ is the actual value, $\hat{y}_i$ is the predicted value from the model, $\bar{y}$ is the mean of the
actual values, and $N$ is the total number of observations.

### Examples

Temperature Estimation:

| Ground Truth Temperature (°C) | Predicted Temperature (°C) |
| ----------------------------- | -------------------------- |
| 25                            | 27                         |
| 35                            | 30                         |

$$
\begin{align}
\text{RMSE} &= \sqrt{\frac{(25 - 27)^2 + (35 - 30)^2}{2}} \\
&= \sqrt{14.5} \\
&\approx 3.81
\end{align}
$$

Age Estimation:

| Ground Truth Age (Years) | Predicted Age (Years) |
| ------------------------ | --------------------- |
| 60                       | 70                    |
| 40                       | 20                    |

$$
\begin{align}
\text{RMSE} &= \sqrt{\frac{(60 - 70)^2 + (40 - 20)^2}{2}} \\
&= \sqrt{250} \\
&\approx 15.8
\end{align}
$$

## Limitations and Biases

While R² is a powerful tool for evaluating the fit of a regression model, it has limitations and can sometimes be
misleading:

- R² always increases when more predictors are added to the model, even if those variables are irrelevant. This can
lead to overfitting where the model performs well on training data but poorly on unseen data.
- A high R² does not imply causation between the independent and dependent variables.
- In cases where the relationship between variables is non-linear, R² may not accurately reflect the model's effectiveness.

To address these issues, it's often recommended to look at adjusted R², which adjusts for the number of predictors in
the model, or to use other metrics in conjunction with R² for a more comprehensive evaluation.

Therefore, while R² can provide valuable insights into the fit of a model, it's essential to consider its limitations
and use it alongside other metrics and validation techniques to thoroughly assess model performance.
