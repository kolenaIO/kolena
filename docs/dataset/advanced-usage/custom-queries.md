---
icon: kolena/developer-16
---

## Query Operators

| **Category**          | **Operators**     | **Example**          |
|-----------------------|-------------------|----------------------|
| **Logical Operators** | `and`             | `@datapoint.a > 2 and @datapoint.b < 3`    |
|                       | `or`              | `@datapoint.a <= 4 or @datapoint.b >= 5`     |
| **Relational Operators** | `==`            | `@datapoint.a == 10`  |
|                       | `!=`              | `@datapoint.a != 5`   |
|                       | `>`               | `@datapoint.a > 20`   |
|                       | `>=`              | `@datapoint.a >= 15`  |
|                       | `<`               | `@datapoint.a < 30`   |
|                       | `<=`              | `@datapoint.a <= 25`  |
| **Arithmetic Operators** | `+`            | `@datapoint.a + 5`    |
|                       | `-`               | `@datapoint.a - 3`    |
|                       | `*`               | `@datapoint.a * 2`    |
|                       | `/`               | `@datapoint.a / 4`    |
| **Power Operator**    | `^`               | `@datapoint.a ^ 2`    |

## Query Functions

| **Function** | **Description**                  | **Example**                          |
|--------------|----------------------------------|--------------------------------------|
| `abs`        | Returns the absolute value       | `abs(@datapoint.a - 10)`            |
| `sqrt`       | Returns the square root value    | `sqrt(@datapoint.a)`                |

## Detailed Description

### Logical Operators

Logical operators are used to combine multiple conditions. The supported logical operators are:

- `and`: Logical AND
- `or`: Logical OR

### Relational Operators

Relational operators are used to compare two values. The supported relational operators are:

- `==`: Equal to
- `!=`: Not equal to
- `>`: Greater than
- `>=`: Greater than or equal to
- `<`: Less than
- `<=`: Less than or equal to

### Arithmetic Operators

Arithmetic operators are used to perform basic mathematical operations. The supported arithmetic operators are:

- `+`: Addition
- `-`: Subtraction
- `*`: Multiplication
- `/`: Division

### Power Operator

The power operator is used to raise a number to the power of another number.

- `^`: Power

### Functions

Functions are pre-defined operations that can be performed on data points. The supported functions are:

- `abs(expression)`: Returns the absolute value of the expression.
- `sqrt(expression)`: Returns the square root of the expression.

### Examples

#### Combining Logical and Relational Operators

```dsl
@datapoint.a > 10 and @datapoint.b < 20

abs(@resultA.recall - @resultB.recall) >= 0.2
```

#### Using Arithmetic and Power Operators

```dsl
@datapoint.a * @datapoint.b + @datapoint.c ^ 2
```

#### Using Functions in Expressions

```dsl
abs(@datapoint.a - 10)

sqrt(@datapoint.b + @datapoint.c)
```
