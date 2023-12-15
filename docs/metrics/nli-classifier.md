# NLI Classification

The [Cross-Encoder for Natural Language Inference](https://huggingface.co/cross-encoder/nli-deberta-v3-base) (NLI) is an
text classification model that takes a pair of text and assigns a label: `'contradiction'`, `'entailment'` or
`'neutral'`. This is useful for hallucination detection, as factual consistency implies the absence of contradictions.
For any label, the higher the score, the more confident the model is to assign that label. So, it chooses to assign the
respective label having the highest score.

## Implementation Details

This cross-encoder for NLI classification is based on Microsoft's
[deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base), which was trained on NLI data, using
[`sentence_transformers.cross_encoder`](https://www.sbert.net/docs/package_reference/cross_encoder.html).

Further details for development and usage can be found on Hugging Face:
[cross-encoder/nli-deberta-v3-base](https://huggingface.co/vectara/hallucination_evaluation_model).
Below is a quick example of how it can be used:

```py
from sentence_transformers import CrossEncoder
nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')

def compute_metric(ground_truth: str, inference: str) -> float:
    scores = nli_model.predict([ground_truth, inference])
    label = ['contradiction', 'entailment', 'neutral'][scores.argmax()]
    return {
        'label': label,
        'contradiction': scores[0],
        'entailment': scores[1],
        'neutral': scores[2],
    }

print(compute_metric("The duck crossed the road", "The duck did not cross the road"))
# Outputs:
# {'label': 'contradiction', 'contradiction': 7.436936, 'entailment': -4.0519376, 'neutral': -3.030173}
```

## Examples
| Ground Truth | Inference | Classification | Contradiction | Entailment | Neutral |
| --- | --- | --- | --- | --- | --- |
| `The duck crossed the road` | `The duck did not cross the road` | `contradiction` | `7.437` | `-4.052` | `-3.030` |
| `The duck crossed the road` | `The animal crossed the road` | `entailment` | `-4.733` | `3.589` | `0.081` |
| `The duck crossed the road` | `The duck crossed my path` | `neutral` | `-2.300` | `-0.173` | `2.059` |

## Limitations and Advantages

1. The model provided on Hugging Face suggests that expected input is simply pairs of text. This means that lengthy
context cannot be provided. If relevant, context can be added to the front of both texts within the pair. For example:
`("What did the duck do? The duck crossed the road.", "What did the duck do? The animal crossed the road.")`.

3. Explainability is a challenge for these scores. This model is a black-box that is not able to explain the reasons
for assigning a contradiction score of `3.0` or `4.5`. Users should examine this classifier's behavior on their own
data, and see how extra details within a ground truth or inference might impact the decision of assigning
`'contradiction'`, `'entailment'` or `'neutral'`

Overall, this NLI classifier is quick and easy to use, and provides insightful metrics relevant to a hallucination
detection system.
