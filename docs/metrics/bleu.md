# BLEU (**B**i**L**ingual **E**valuation **U**nderstudy) Score

The BLEU score is a metric commonly used in a variety of NLP tasks ranging from Machine Translation to Text Summarization, to evaluate the quality of machine-generated text. It quantifies the similarity between the generated text and ground truth text as a score between 0 and 1 — 0 meaning there is no overlap with the ground truth text, and 1 meaning there is a perfect overlap. As the name suggests, it was originally developed for evaluating machine-translation models, but has since been adapted to many different tasks within NLP due to its dynamic nature for measuring textual similarity.

## Implementation Details
### Definition
The BLEU score is mathematically defined as:

$$\begin{align*} \text{BLEU} &= \text{Brevity Penalty} \times \text{n-Gram Overlap} \\
&= \min\left(1, \exp\left(1 - \frac{\text{reference length}}{\text{output length}}\right)\right) \times \left(\prod_{i=1}^{4}\text{i-Gram Precision}^{1/4}\right)
\end{align*}$$

where the i-Gram precision is calculated as:

$$
p_i = \frac{\text{Clipped} \text{ count of matching i-grams in machine-generated text}^1}{\text{Total number of i-grams in machine-generated text}}
$$

Below, we explain the significance of the two components in the BLEU formula.
??? info "n-Gram Overlap"
    The **n-Gram Overlap** counts the number of 1-grams, 2-grams, 3-grams, and 4-grams of the output text that match the 1-, ..., 4-grams in the reference text — which is analogous to a precision score for the text. The 1-gram precision ensures that the correct vocabulary is used, whereas the 4-gram precision ensures that the generated text is coherent.
??? info "Brevity Penalty"
    The **Brevity Penalty** is also applied to penalize the score for generating sentences that are less in length than the reference text. This is due to the fact that the n-Gram Overlap precision tends to give disproportionately high values to generated texts that are very short in length, but contain most of the n-grams in the reference text.

<div class="footnote-content">
    <p style="font-size: smaller;">
        <sup>1</sup> The <i>clipped count of matching i-grams in machine-generated text</i> is the minimum between the count of i-grams in the machine-generated text and the maximum count of i-grams in any of the reference texts for a given i-gram.
    </p>
</div>

### Interpretation
A known fact about BLEU scores is that they are not to be compared between different workflows and tasks. An excellent BLEU score for one task may be subpar for another. The score only serves as a general guideline to quantify the performance of your model — not to replace human judgement. That being said, what do these scores really mean, and how can we decipher them?

1. **Higher is better**: Though BLEU scores may vary among tasks, one thing is for sure - higher scores are better. Generally speaking, a commonly accepted guideline is as follows: <br>

    | Score | Qualitative Interpretation |
    | ---   | ---                        |
    | < 0.1 | Useless output             |
    | 0.1 - 0.4 | Varies in quality; May not be acceptable |
    | 0.4 - 0.6 | High quality generated text |
    | > 0.6 | Better than human quality  |

2. **Track Trends Over Time**: Rising scores signal improvements in models, while drops could hint at issues or changes in your dataset.

3. **Combine It With Other Metrics**: BLEU primarily measures n-gram overlap, overlooking some nuancies like context and understanding. While a high BLEU score is promising, it doesn't guarantee flawless text. A complementary metric like [BertScore](bertscore.md) may help in quantifying your model's performance from other perspectives.

## Example

**Generated Sentence**: Kolena is an ML testing and debugging platform to find hidden model behaviors and demystify model development <br>
**Reference Sentence**: Kolena is a comprehensive machine learning testing and debugging platform to surface hidden model behaviors and take the mystery out of model development

??? example "Step 1: Tokenization & n-Grams"
    Splitting our sentences up into 1-, ..., 4-grams, we get:

    **Generated Sentence**:

    | n | n-Grams |
    | ---   | ---                        |
    | 1 | ["kolena", "is", "an", "ml", "testing", "and", "debugging", "platform", "to", "find", "hidden", "model", "behaviors", "and", "demystify", "model", "development"]|
    | 2 | ["kolena is", "is an", "an ml", "ml testing", "testing and", "and debugging", "debugging platform", "platform to", "to find", "find hidden", "hidden model", "model behaviors", "behaviors and", "and demystify", "demystify model", "model development"]|
    | 3 | ["kolena is an", "is an ml", "an ml testing", "ml testing and", "testing and debugging", "and debugging platform", "debugging platform to", "platform to find", "to find hidden", "find hidden model", "hidden model behaviors", "model behaviors and", "behaviors and demystify", "and demystify model", "demystify model development"] |
    | 4 | ["kolena is an ml", "is an ml testing", "an ml testing and", "ml testing and debugging", "testing and debugging platform", "and debugging platform to", "debugging platform to find", "platform to find hidden", "to find hidden model", "find hidden model behaviors", "hidden model behaviors and", "model behaviors and demystify", "behaviors and demystify model", "and demystify model development"]  |

    **Reference Sentence**:

    | n | n-Grams |
    | ---   | ---                        |
    | 1 | ["kolena", "is", "a", "comprehensive", "machine", "learning", "testing", "and", "debugging", "platform", "to", "surface", "hidden", "model", "behaviors", "and", "take", "the", "mystery", "out"]|
    |...|
    | 4 | ["kolena is a comprehensive", "is a comprehensive machine", "a comprehensive machine learning", "comprehensive machine learning testing", "machine learning testing and", "learning testing and debugging", "testing and debugging platform", "and debugging platform to", "debugging platform to surface", "platform to surface hidden", "to surface hidden model", "surface hidden model behaviors", "hidden model behaviors and", "model behaviors and take", "behaviors and take the", "and take the mystery", "take the mystery out", "the mystery out of", "mystery out of model", "out of model development"]  |

??? example "Step 2: Calculate n-Gram Overlap"
    Next, lets calculate the clipped precision scores for each of the n-Grams. Recall that the precision formula is:

    $$
    p_i = \frac{\text{Clipped} \text{ count of matching i-grams in machine-generated text}^1}{\text{Total number of i-grams in machine-generated text}}
    $$

    <center>

    | n | Clipped Precision  |
    | --- | --- |
    | 1 | 14 / 17 = 0.824 |
    | 2 | 10 / 16 = 0.625 |
    | 3 | 6 / 15 = 0.400 |
    | 4 | 3 / 14 = 0.214 |

    </center>

    So, our n-Gram overlap is:

    $$
    0.824^{0.25}\cdot0.625^{0.25}\cdot0.400^{0.25}\cdot0.214^{0.25} = 0.458
    $$

??? example "Step 3: Calculate Brevity Penalty"
    We apply a brevity penalty to prevent the BLEU score from giving undeservingly high scores for short generated texts. Recall that our formula is:

    $$
    \min\left(1, \exp\left(1 - \frac{\text{reference length}}{\text{output length}}\right)\right)
    = \min\left(1, \exp\left(1 - \frac{\text{23}}{\text{17}}\right)\right)
    = 0.703
    $$

??? example "Step 4: Calculate BLEU"
    Combining our n-Gram overlap and Brevity Penalty, our final BLEU score is:

    $$
    \text{BLEU} = \text{Brevity Penalty} \times \text{n-Gram Overlap} = 0.458 \times 0.703 = 0.322
    $$

    Note that in most cases, we may take the average of the BLEU score with respect to multiple reference texts - since multiple interpretations of the same sentences can be allowed. For example, if we calculated the BLEU score with the reference text being "Kolena is a comprehensive machine learning testing and debugging platform to surface hidden model behaviors and demystify model development", our BLEU score would be 0.571 - a much higher score from a small semantic change.


## Advantages and Limitations
Though BLEU is a popular metric in NLP workflows, it comes with its limitations.

1. BLEU fails to consider the semantics of texts. As seen in the example, simply changing "take the mystery out of" to "demistify" — while the text still retains the exact same meaning — yields a much better score, going from 0.3 to 0.6. In contrast with embeddings-based metrics like BertScore, n-gram-based metrics like BLEU only consider the words in the generated text, rather than the meaning. However, this is addressed by providing multiple possible reference texts when calculating the BLEU score.
2. BLEU does not consider the order of words. Comparing "The quick brown fox jumps over the lazy dog" with "Brown jumps the dog over quick the fox lazy" would yield a perfect score of 1.0 despite the generated-text having zero meaning. Similarly, BLEU does not consider the importance of words in a sentence either. It weighs unimportant words like "the", "an", "too", as much as it would the nouns and verbs in the sentence. Once again, these pitfalls are addressed by embeddings-based metrics like BertScore.

That being said, the metric still has its advantages. It is quick and easy to compute, as opposed to other metrics like BertScore which would take significantly longer to compute and is not easy to justify. Furthermore, it is relatively similar to human judgement (as seen in the [figure below](https://aclanthology.org/P02-1040.pdf)), and is commonly used within NLP which allows you to easily benchmark your models with others and identify pain points.

[![BLEU Judgement Image](../assets/images/bleu-judgement.png)](https://aclanthology.org/P02-1040.pdf)
