---
icon: kolena/metrics-glossary-16
hide:
  - toc
---

# :kolena-metrics-glossary-20: Metrics Glossary

This section contains guides for different metrics used to measure model performance.

Each ML use case requires different metrics. Using the right metrics is critical for understanding and meaningfully
comparing model performance. In each metrics guide, you can learn about the metric with examples, its limitations and
biases, and its intended uses.

<div class="grid cards" markdown>

- [Accuracy](accuracy.md)

    ---

    Accuracy measures how well a model predicts correctly. It's a good metric for assessing model performance in simple
    cases with balanced data.

- [Average Precision (AP)](average-precision.md)

    ---

    Average precision summarizes a precision-recall (PR) curve into a single threshold-independent value
    representing model's performance across all thresholds.

- [Averaging Methods: Macro, Micro, Weighted](averaging-methods.md)

    ---

    Different averaging methods for aggregating metrics for **multiclass** workflows, such as classification and
    object detection.

- [Confusion Matrix](confusion-matrix.md)

    ---

    Confusion matrix is a structured plot describing classification model performance as a table that highlights counts
    of objects with predicted classes (columns) against the actual classes (rows), indicating how confused a model is.

- [F<sub>1</sub>-score](f1-score.md)

    ---

    F<sub>1</sub>-score is a metric that combines two competing metrics, [precision](precision.md) and
    [recall](recall.md) with an equal weight. It symmetrically represents both precision and recall as one metric.

- [False Positive Rate (FPR)](fpr.md)

    ---

    False positive rate (FPR) measures the proportion of negative ground truths that a model incorrectly predicts as
    positive, ranging from 0 to 1. It is useful when the objective is to measure and reduce false positive inferences.

- [Precision](precision.md)

    ---

    Precision measures the proportion of positive inferences from a model that are correct. It is useful when the
    objective is to measure and reduce false positive inferences.

- [Precision-Recall (PR) Curve](pr-curve.md)

    ---

    Precision-recall curve is a plot that gauges machine learning model performance by using [precision](precision.md)
    and [recall](recall.md). It is built with precision on the y-axis and recall on the x-axis computed across many
    thresholds.

- [Recall (TPR, Sensitivity)](recall.md)

    ---

    Recall, also known as true positive rate (TPR) and sensitivity, measures the proportion of all positive ground
    truths that a model correctly predicts. It is useful when the objective is to measure and reduce false negative
    ground truths, i.e. model misses.

- [Receiver Operating Characteristic (ROC) Curve](roc-curve.md)

    ---

    A receiver operating characteristic (ROC) curve is a plot that is used to evaluate the performance of binary
    classification models by using the [true positive rate (TPR)](./recall.md) and the
    [false positive rate (FPR)](./fpr.md).

- [Specificity (TNR)](specificity.md)

    ---

    Specificity, also known as true negative rate (TNR), measures the proportion of negative ground truths that a
    model correctly predicts, ranging from 0 to 1. It is useful when the objective is to measure the model's ability to
    correctly identify the negative class instances.

- [TP / FP / FN / TN](tp-fp-fn-tn.md)

    ---

    The counts of TP, FP, FN and TN ground truths and inferences are essential for summarizing model performance. They
    are the building blocks of many other metrics, including [accuracy](accuracy.md), [precision](precision.md),
    and [recall](recall.md).

</div>

## Computer Vision

<div class="grid cards" markdown>

- [Geometry Matching](geometry-matching.md)

    ---

    Geometry matching is the process of matching inferences to ground truths for computer vision workflows with a
    localization component. It is a core building block for metrics such as [TP, FP, and FN](tp-fp-fn-tn.md), and any
    metrics built on top of these, like [precision](precision.md), [recall](recall.md), and
    [F<sub>1</sub>-score](f1-score.md).

- [Intersection over Union (IoU)](iou.md)

    ---

    IoU measures overlap between two geometries, segmentation masks, sets of labels, or time-series snippets.
    Also known as Jaccard index in classification workflow.

</div>

## Natural Language Processing

<div class="grid cards" markdown>

- [BERTScore](bertscore.md)

    ---

    BERTScore is a metric used in NLP workflows to measure textual similarity between candidate texts and reference
    texts.

- [BLEU](bleu.md)

    ---

    BLEU is a metric commonly used in a variety of NLP workflows to evaluate the quality of candidate texts. BLEU can be
    thought of as an analog to [precision](precision.md) for text comparisons.

- [METEOR](meteor.md)

    ---

    METEOR is a widely recognized and vital metric used in NLP. It is used to measure the quality of candidate texts
    against reference texts. Though it is an n-gram based metric, it goes beyond traditional methods by factoring in
    elements such as precision, recall, and order to provide a comprehensive measure of text quality.

- [Perplexity](perplexity.md)

    ---

    Perplexity is a metric commonly used in natural language processing to evaluate the quality of language models,
    particularly in the context of text generation. Unlike metrics such as BLEU or BERT, perplexity doesn't directly
    measure the quality of generated text by comparing it with reference texts. Instead, perplexity assesses the
    "confidence" or "surprise" of a language model in predicting the next word in a sequence of words.


- [ROUGE-N](rouge-n.md)

    ---

    ROUGE-N, a metric within the broader ROUGE metric collection, is a vital metric in the field of NLP. It assesses
    the quality of a candidate text by measuring the overlap of n-grams between the candidate text and reference texts.
    ROUGE-N can be thought of as an analog to [recall](recall.md) for text comparisons.

</div>

## Large Language Models

This section of the glossary contains LLM evaluation metrics to identify and measure the number of hallucinations produced by a generative language model. A hallucination is an output by a model that is made up and/or non-factual.

### Risks of Hallucinations
In a lower-risk setting, language model hallucinations can be something as simple as getting the birth date of a celebrity wrong which while isn not ideal is not inherently harmful either. However, in higher-risk and real-world situations, generative models that consistently mix facts and fiction can contribute to the spread of misinformation and mislead individuals.

<div class="grid cards" markdown>

- [GPT4 Prompt Engineering](gpt4-prompt-eng.md)

    ---

    This technique involves prompt engineering with the latest state-of-the-art model for LLM evaluation, GPT4. Methods like chain-of-thought and self-consistency are employed to determine whether or not a prompt contains a hallucination.

- [SelfCheckGPT](selfcheckgpt.md)

    ---

    SelfCheckGPT is a sampling-based approach that samples using the same prompt to determine whether or not the model is consistent with itself. The prompt is considered a hallucination if sampled responses that contradict one another or are not consistent.

</div>
