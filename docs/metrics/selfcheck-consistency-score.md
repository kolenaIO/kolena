# SelfCheck Consistency Score

SelfCheck Consistency Score is a simple sampling-based technique that relies on the idea that an
Language Language Model (LLM) has inherent knowledge of facts and if prompted multiple times should be able to output
similar or consistent responses which can be used to score the LLM. Existing fact-checking methods require output
token-level probabilities which may not be accessible. As such, [SelfCheckGPT](https://arxiv.org/abs/2303.08896)
presents a black-box zero-resource solution that only requires text-based responses to evaluate for hallucinations.

#### When can this metric be used?

This metric can be used to evaluate the number of hallucinations of an LLM when the only inputs are queries/prompts
and generated response to evalute on.

#### How is it computed?

Here are three variants from SelfCheckGPT tool to measure consistency:

1. SelfCheckGPT with BERTScore
2. SelfCheckGPT with $n$-gram
3. SelfCheckGPT with LLM Prompt

For SelfCheckGPT with BERTScore, values are in the range of `[0.0, 1.0]` with a higher value suggesting that the response may
be non-factual.

For SelfCheckGPT with $n$-gram, values are in the range of `[0.0, +inf)` with a higher value suggesting that the response may
be non-factual.

For SelfCheckGPT with LLM Prompt, values are either `Yes` or `No` where `Yes` implies that the responses are consistent
(no hallucination)


## Implementation Details

SelfCheckGPT contains detailed and comprehensive usage details in their [code repository](https://github.com/potsawee/selfcheckgpt).
However, we present some simple example use cases for each variant below using an example from the [TruthfulQA](https://github.com/sylinrl/TruthfulQA) dataset

```
Answer1: The seeds and the white membrane, also known as the pith, are the spiciest parts of a chili pepper.
Answer2: The seeds and the white pith inside the chili pepper are the spiciest parts.
Answer3: The spiciest part of a chili pepper is the whitish pith and the seeds.
Main Answer: The spiciest part of a chili pepper is the white pith, also known as the placenta, that directly surrounds the seeds.
```

!!! info "SelfCheck Tool Installation"

    ```
    pip install selfcheckgpt
    ```
    For further details, checkout [SelfCheckGPT](https://github.com/potsawee/selfcheckgpt).

### BERTScore-based Consistency Score

Given the $i$-th sentence of a response $R$, SelfCheckGPT with BERTScore finds the average BERTScore with the most
similar sentence from each of the $N$ sample responses. The idea is that if the same information appears in multiple
sample responses then it should be factually correct.

!!! info "Guide: BERTScore"

    Read the [BERTScore](./bertscore.md) guide if you're not familiar.


#### Code Example

The score we compute for each sentence will be between `[0.0, 1.0]` with a higher value suggesting that the response may
be non-factual.

```py
from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore

selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True)

sent_scores_bertscore = selfcheck_bertscore.predict(
    sentences = [main_answer],
    sampled_passages = [answer1, answer1, answer3],
)
```

```
[0.31013352]
```

### $n$-Gram-based Consistency Score

SelfCheckGPT with $n$-gram trains a simple $n$-gram model using the $N$ sample responses as well as the response $R$ to
compute token-level probabilities for tokens in each sentence from $R$. The score is computed by the following formula:

$$
S_{n\text{-gram}}^\text{Avg}(i) = -\frac{1}{J} \sum_{j} \log{\tilde{p}_{ij}}
$$

where $\tilde{p}_{ij}$ is the probability output from the $n$-gram model of the $j$-th token from $i$-th sentence in $R$.

#### Code Example

The scores we compute are at sentence- and document-level where values lie between `[0.0, +inf)` with a higher value
suggesting that the response may be non-factual.

```py
from selfcheckgpt.modeling_selfcheck import SelfCheckNgram

selfcheck_ngram = SelfCheckNgram(n=1)

sent_scores_ngram = selfcheck_ngram.predict(
    sentences = [main_answer],
    passage = main_answer,
    sampled_passages = [answer1, answer1, answer3],
)
print(sent_scores_ngram)
```

```
{'sent_level': { # sentence-level score similar to MQAG and BERTScore variant
    'avg_neg_logprob': [3.1152231849792478],
    'max_neg_logprob': [4.418840607796598]
    },
'doc_level': { # document-level score such that avg_neg_logprob is computed over all tokens
    'avg_neg_logprob': 3.1152231849792478,
    'avg_max_neg_logprob': 4.418840607796598
    }
}
```


### LLM Prompt-based Consistency Score

With this approach, SelfCheckGPT utilizes an LLM such as `gpt-3.5-turbo` and a prompt to compute a boolean consistency
score where `Yes` implies that the context and sentence are consistent (no hallucination) and otherwise, `No`, for each sample
answer which is passed as the context in our prompts.

#### Code Example

We define a `PROMPT`

```
Context: {answer}
Sentence: {main_answer}
Is the sentence supported by the context above?
Answer Yes or No. If the answer is No, on the next line, explain in about ten words why it is not supported by the context.
```

Then, using OpenAI's API we make a query

```py
response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "user",
        "content": PROMPT
        },
    ],
    temperature=0,
    max_tokens=50,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)
```

after pre-processing we can expect output like the below

```
No
The sentence is not supported by the context because it states that the spiciest part of a chili pepper is the white pith,
also known as the placenta, which is not mentioned in the context.
```

To obtain a numerical consistency score we perform the same operation but on different context answers and aggregate across the
returned boolean consistency scores.

## Limitations and Advantages

SelfCheckGPT is a very powerful black-box evaluation approach thats advantages come from the fact that it does not require labeled
data to compute a consistency score. However, SelfCheckGPT can be quite limiting when evaluating sentences that contain both factual
and non-factual statements. In addition, for the LLM Prompt-based Consistency Score it contains the same limitations as
[LLM Prompt-based Metric](./llm-prompt-based-metric.md).
