---
icon: kolena/developer-16
---

# :kolena-developer-20: Custom Queries and Fields

Custom queries allow you to filter and sort your data flexibly and find datapoints of interest faster.
The custom query function can be accessed from the filter and sort components.

<figure markdown>
![Access Custom Queries](../../assets/images/custom-queries-dark.gif#only-dark)
![Access Custom Queries](../../assets/images/custom-queries-light.gif#only-light)
<figcaption>Access Custom Queries</figcaption>
</figure>

Custom fields allow you configure new fields based on existing fields using common operations.

<figure markdown>
![Access Custom Fields](../../assets/images/custom-fields-dark.gif#only-dark)
![Access Custom Fields](../../assets/images/custom-fields-light.gif#only-light)
<figcaption>Access Custom Fields</figcaption>
</figure>

You can access datapoint fields by typing `@datapoint.` or result fields via `@result.`.

## Details

This table summarized the the available operations.

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
|                          | `floor`       | `floor(@datapoint.a)`                    | Returns the value rounded to the nearest equal or smaller integer |
|                          | `round`       | `round(@datapoint.a)`                    | Returns the rounded value                                         |
|                          | `sqrt`        | `sqrt(@datapoint.a)`                     | Returns the square root value                                     |
| **Array Functions**      | `array_size`  | `array_size(@datapoint.a[])`             | Returns the number of objects in the array                        |
|                          | `filter`      | `filter(@datapoint.a, val -> val.area >= 400)`| Returns an array with objects that have an area of more than 400|

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

    **Using array filters and size**

    This function returns the number of objects that have an area of more than 400
    ```dsl
    array_size(filter(@datapoint.b[], val -> val.area >= 400))
    ```
