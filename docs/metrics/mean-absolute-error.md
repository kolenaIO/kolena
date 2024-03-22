# Mean Absolute Error

Mean Absolute Error (MAE) is a popular metric used in assessing regression model performance
for its overall simplicity and interpretability.

MAE measures the average magnitude of errors in predictions, without considering their direction.
It represents the mean of the absolute differences between predicted and actual values across a dataset,
treating each discrepancy equally.

## Implementation Details

MAE is calculated by taking the average of the absolute differences between the predicted values and the actual values.
This can be mathematically represented as:

$$
\frac{1}{N} \sum_{i=1}^{N}|x_i-y_i|
$$

where $x$ is the numerical value from the actual values, and $y$ is the corresponding numerical value from the
predicted values for a total of $N$ number of predictions.

### Examples

Temperature Estimation:

<div class="grid" markdown>
| Ground Truth Temperature (&deg;C) | Predicted Temperature (&deg;C) |
| --- | --- |
| 25 | 27 |
| 35 | 30 |

$$
\begin{align}
\text{MAE} &= \frac{|25 - 27| + |35 - 30|}{2} \\[1em]
&= 3.5
\end{align}
$$
</div>

Age Estimation::

<div class="grid" markdown>
| Ground Truth Age (Years) | Predicted Age (Years) |
| --- | --- |
| 60 | 70 |
| 40 | 20 |

$$
\begin{align}
\text{MAE} &= \frac{|60 - 70| + |40 - 20|}{2} \\[1em]
&= 15
\end{align}
$$
</div>

## Limitations and Biases

While Mean Absolute Error (MAE) is straightforward to interpret, it treats all errors equally,
which might not reflect the true impact of outliers or extreme errors on the model's performance.

In scenarios where the dataset contains outliers or extreme values, MAE might not accurately represent
the model's overall predictive capability in a wholistic manner. It's important to complement MAE with other
evaluation metrics, such as Root Mean Squared Error (RMSE), which penalizes larger errors more heavily.

Thus, while MAE provides valuable insights into prediction accuracy, it's advisable to consider it
alongside other metrics for a comprehensive evaluation of regression models.
