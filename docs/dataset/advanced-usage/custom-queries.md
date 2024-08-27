---
icon: kolena/developer-16
---

# :kolena-developer-20: Custom Queries

Custom queries allow you to filter and sort your data flexibly and find datapoints of interest faster.
The custom query function can be accessed from the filter and sort components.

<figure markdown>
![Access Custom Queries](../../assets/images/custom-queries-dark.gif#only-dark)
![Access Custom Queries](../../assets/images/custom-queries-light.gif#only-light)
<figcaption>Access Custom Queries</figcaption>
</figure>

You can access datapoint fields by typing `@datapoint.` or result fields via `@result.`.

## Custom Query Details

This table summarized the custom query capabilities.

| **Category**             | **Operators** | **Example**                              | **Description**                                                   |
|--------------------------|---------------|------------------------------------------|-------------------------------------------------------------------|
| **Logical Operators**    | `and`         | `@datapoint.a > 2 and @datapoint.b < 3`  | Logical AND                                                       |
|                          | `or`          | `@datapoint.a <= 4 or @datapoint.b >= 5` | Logical OR                                                        |
| **Relational Operators** | `==`          | `@datapoint.a == 10`                     | Equal to                                                          |
|                          | `!=`          | `@datapoint.a != 5`                      | Not equal to                                                      |
|                          | `>`           | `@datapoint.a > 20`                      | Greater than                                                      |
|                          | `>=`          | `@datapoint.a >= 15`                     | Greater than or equal to                                          |
|                          | `<`           | `@datapoint.a < 30`                      | Less than                                                         |
|                          | `<=`          | `@datapoint.a <= 25`                     | Less than or equal to                                             |
| **Arithmetic Operators** | `+`           | `@datapoint.a + 5`                       | Addition                                                          |
|                          | `-`           | `@datapoint.a - 3`                       | Subtraction                                                       |
|                          | `*`           | `@datapoint.a * 2`                       | Multiplication                                                    |
|                          | `/`           | `@datapoint.a / 4`                       | Division                                                          |
| **Power Operator**       | `^`           | `@datapoint.a ^ 2`                       | Power                                                             |
| **Functions**            | `abs`         | `abs(@datapoint.a - 10)`                 | Returns the absolute value                                        |
|                          | `sqrt`        | `sqrt(@datapoint.a)`                     | Returns the square root value                                     |
|                          | `floor`       | `floor(@datapoint.a)`                    | Returns the value rounded to the nearest equal or smaller integer |
|                          | `round`       | `round(@datapoint.a)`                    | Returns the rounded value                                         |

!!! Example
    **Combining Logical and Relational Operators**

    ```dsl
    @datapoint.a > 10 and @datapoint.b < 20
    ```
    ```dsl
    abs(@resultA.recall - @resultB.recall) >= 0.2
    ```

    **Using Arithmetic and Power Operators**

    ```dsl
    @datapoint.a * @datapoint.b + @datapoint.c ^ 2
    ```

    **Using Functions in Expressions**

    ```dsl
    abs(@datapoint.a - 10)
    ```
    ```dsl
    sqrt(@datapoint.b + @datapoint.c)
    ```
