# SelfCheck Consistency Score

SelfCheck Consistency Score is a simple sampling-based technique that relies on the idea that an
Language Language Model (LLM) has inherent knowledge of facts and if prompted multiple times should be able to output
similar or consistent responses which can be used to score the LLM. Existing fact-checking methods require output
token-level probabilities which may not be accessible. As such, [SelfCheckGPT](https://arxiv.org/abs/2303.08896)
presents a black-box zero-resource solution that only requires text-based responses to evaluate for hallucinations.

To compute the SelfCheck Consistency Score, we focus on a few SelfCheckGPT variants:

1. SelfCheckGPT with BERTScore
3. SelfCheckGPT with $n$-gram
5. SelfCheckGPT with LLM Prompt

## Implementation Details

SelfCheckGPT contains detailed and comprehensive usage details in their [code repository](https://github.com/potsawee/selfcheckgpt).
However, we present some simple example use cases for each variant below using an example from the TruthfulQA dataset

```
Answer: The seeds and the white membrane, also known as the pith, are the spiciest parts of a chili pepper.
Main Answer: The spiciest part of a chili pepper is the white pith, also known as the placenta, that directly surrounds the seeds.
```

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
    sampled_passages = [answer],
)
```

```
[0.31471425]
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
    sampled_passages = [answer],
)
print(sent_scores_ngram)
```

```
{'sent_level': { # sentence-level score similar to MQAG and BERTScore variant
    'avg_neg_logprob': [2.9464540757764373],
    'max_neg_logprob': [3.828641396489095]
    },
'doc_level': { # document-level score such that avg_neg_logprob is computed over all tokens
    'avg_neg_logprob': 2.9464540757764373,
    'avg_max_neg_logprob': 3.828641396489095
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

SelfCheckGPT is a very powerful black-box evaluation approach thats advantages come from its overall lower resource usage as
well as its performance despite only having access to text-based responses. However, SelfCheckGPT can often be mislead by sentences
that contain consistent by factually incorrect statements and for the best variant of SelfCheckGPT with LLM Prompt, it can also be
computationally heavy.
