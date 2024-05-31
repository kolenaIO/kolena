# Pearson Correlation Coefficient

Pearson's correlation coefficient is a statistical measure that evaluates the strength and direction of the linear
relationship between two continuous variables. The values range from -1 to +1. A value close to +1 indicates a strong
positive linear correlation, where as one variable increases, the other also increases proportionally. A value close
to -1 signifies a strong negative linear correlation, meaning as one variable increases, the other decreases
proportionally. A value of 0 indicates no linear correlation between the variables.

## Implementation Details

The correlation coefficient is calculated by dividing the covariance of $x$ and $y$ by their individual standard
deviations. This can be mathematically represented as:

$$
r = \frac{\text{cov}(x, y)}{\sigma_x \sigma_y} = \frac{\sum_{i=1}^{N} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{N}
(x_i - \bar{x})^2 \sum_{i=1}^{N} (y_i - \bar{y})^2}}
$$

where $x$ and $y$ are the collection of data points.

### Examples

Temperature Correlation:

<div class="grid" markdown>

| Ground Truth Temperature (&deg;C) | Predicted Temperature (&deg;C) |
| --- | --- |
| 25  | 30  |
| 35  | 28  |
| 30  | 27  |
| 40  | 35  |

</div>

$$
\begin{align}
\bar{x} &= \frac{25 + 35 + 30 + 40}{4} = 32.5 \\[1em]
\bar{y} &= \frac{30 + 28 + 27 + 35}{4} = 30 \\[1em]
r &= \frac{\sum_{i=1}^{4} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{4} (x_i - \bar{x})^2 \sum_{i=1}^{4} (y_i -
\bar{y})^2}} \\[1em]
&= \frac{(25 - 32.5)(30 - 30) + (35 - 32.5)(28 - 30) + (30 - 32.5)(27 - 30) +
(40 - 32.5)(35 - 30)}{\sqrt{[(25 - 32.5)^2 +
(35 - 32.5)^2 + (30 - 32.5)^2 + (40 - 32.5)^2] [(30 - 30)^2 + (28 - 30)^2 + (27 - 30)^2 + (35 - 30)^2]}} \\[1em]
&\approx 0.58
\end{align}
$$

## Limitations and Biases

Pearson's correlation coefficient is useful for measuring the strength and direction of linear relationships between
variables. However, it assumes linearity and is sensitive to outliers, which can distort the results. In datasets with
non-linear relationships or significant outliers, Pearson's correlation coefficient may not provide an accurate
representation.
