# SelfCheckGPT

[SelfCheckGPT](https://arxiv.org/abs/2303.08896) is a simple sampling-based technique that relies on the idea that an
Language Language Model (LLM) has inherent knowledge of facts and if prompted multiple times should be output a
similar/consistent response. Existing fact-checking methods require output token-level probabilities which may not be
accessible. As such, [SelfCheckGPT](https://arxiv.org/abs/2303.08896) presents a black-box zero-resource solution that
only requires text-based responses to evaluate for hallucinations.

![SelfCheckGPT Image](../assets/images/selfcheck_qa_prompt.png)
*[Image from SelfCheckGPT repo](https://github.com/potsawee/selfcheckgpt)*

SelfCheckGPT has a few variants:

1. SelfCheckGPT with BERTScore
2. SelfCheckGPT with Question Answering
3. SelfCheckGPT with $n$-gram
4. SelfCheckGPT with Natural Language Inference (NLI)
5. SelfCheckGPT with LLM Prompt

For each user query, a response is generated from the LLM that we are trying to evaluate, the response is referred to as
$R$. Using the same query, $N$ further response samples are generated. For each approach below, SelfCheckGPT predicts a
score for the $i$-th sentence that is between `0` and `1` where `0` represents consistent grounded information while `1.0`
implies that $R$ is hallucinated.

## Implementation Details

SelfCheckGPT contains detailed and comprehensive usage details in their [code repository](https://github.com/potsawee/selfcheckgpt).

### BERTScore-based Consistency Score

Given the $i$-th sentence of a response $R$, SelfCheckGPT with BERTScore finds the average BERTScore with the most
similar sentence from each of the $N$ sample responses. The idea is that if the same information appears in multiple
sample responses then it should be factually correct.

!!! info "Guide: BERTScore"

    Read the [BERTScore](./bertscore.md) guide if you're not familiar.

### $n$-Gram-based Consistency Score

SelfCheckGPT with $n$-gram trains a simple $n$-gram model using the $N$ sample responses as well as the response $R$ to
compute token-level probabilities for tokens in each sentence from $R$. The score is computed by the following formula:

$$
S_{n\text{-gram}}^\text{Avg}(i) = -\frac{1}{J} \sum_{j} \log{\tilde{p}_{ij}}
$$

where $\tilde{p}_{ij}$ is the probability output from the $n$-gram model of the $j$-th token from $i$-th sentence in $R$.

### LLM Prompt-based Consistency Score

With this approach, SelfCheckGPT utilizes an LLM such as GPT-3 and the following prompt schema:

```
Context: {}
Sentence: {}
Is the sentence supported by the context above?
Answer Yes or No:
```

to obtain a hallucination score where `Yes` is `0.0` and `No` is `1.0`.

## Limitations and Advantages

SelfCheckGPT is a very powerful black-box evaluation approach thats advantages come from its overall lower resource usage as
well as its performance despite only having access to text-based responses. However, SelfCheckGPT can often be mislead by sentences
that contain consistent by factually incorrect statements and for the best variant of SelfCheckGPT with LLM Prompt, it can also be
computationally heavy.
