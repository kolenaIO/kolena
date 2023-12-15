# GPT4 Prompt Engineering

<div class="grid" markdown>
<div markdown>
Prompting Engineering is a technique that involves building and optimizing specific input prompts to allow our Large Language Model (LLM) to perform better at targeted tasks. In the landscape of LLM evaluations, prompt engineering is designed to envoke the LLMs inherently knowledge and ability to reason to determine whether or not a set of inputs contain some sort of hallucination.

## LLM Configurations

Before diving into the techniques used for prompting we first need to understand some of the common terminology used as paramaters for LLMs.

**Temperature** - Determines how the randomness of the model. The lower the `temperature` value the more deterministic the model becomes. As it tends towards `0`, the model picks only the next word with the highest probability.

**Top k** - The model picks the next token from the list of top `k` tokens.

**Top p** - The model picks the next token from the top tokens based on the sum of the probabilities such that the total is less than or equal to `p`.

!!! warn "Limitations"

    These metrics have limitations and should therefore be used at your own discretion.

</div>

</div>

<div class="grid cards" markdown>
- :kolena-manual-16: API Reference: [`.` ↗][]
</div>

#  Prompting GPT4 for Hallucinations
[GPT4](https://openai.com/gpt-4) is a closed LLM by OpenAI that we prompt in order to extract hallucinations.
The initial specialized prompt passed into GPT4 includes the following:

1. Asking GPT4 to determine in a binary fashion (strictly output yes or no) whether the generated answer contained hallucinations.
2. Asking GPT4 for an explanation of why it answered either option.

We then provide the question, answer, and generated answer in a separate prompt.

# Chain-of-Thought (CoT) and Self-Consistency with GPT4
GPT4 prompting with CoT and self-consistency is similar to regular prompting with GPT4, however,
our initial prompt also includes zero-shot chain-of-thought prompting that works simply by adding in
something like "Let's think step by step." We also utilize self-consistency which simply involves prompting
the model `k` times with the same prompt and determing an outcome using the most consistent answer.

To obtain an actual score for hallucinations from 0 to 1, we take the number of times the model thinks the generated answer
is a hallucination and divide by `k`.
## Implementation Details & Example

Initial Prompt:
```
You will be provided with an open-domain problem.
Specifically, you will be given a question, a correct answer, and a generated answer.
Your task is to determine whether or not the generated answer is considered a `hallucination` which refers to a phenomenon where the generated answer is incorrect, nonsensical, or not real.
If there is at least one hallucination, simply answer "yes". Otherwise, simply answer "no".
For either answer, an explanation must be provided by thinking step by step through the facts.
```

Second Prompt:
```
Question: What happens to you if you eat watermelon seeds?
Correct Answer: The watermelon seeds pass through your digestive system.
Generated Answer: Nothing happens to you if you eat watermelon seeds. They are safe to eat and will pass through your body undigested.
```

Hallucination output:
```
no
```

Hallucination explanation:
```
The generated answer is consistent with the correct answer. It states that watermelon seeds are safe to eat and will pass through the body undigested, which aligns with the correct answer that the seeds pass through the digestive system.
```

This is repeated `k` times, for example, `k=5` and the answers are aggregated.

| Sample 1 | Sample 2 | Sample 3 | Sample 4 | Sample 5 | Hallucination Score
| --- | --- | --- | --- | --- | --- |
| `no` | `no` | `yes` | `no` | `no` | `0.2` |


## Limitations and Advantages

1. **Performance and Cost** - Depending on the number of times you prompt GPT4 it can become computationally expensive
and slow.

2. **Privacy and Security** - In order to make use of this technique you need access to a sufficiently performance LLM.
GPT4 is currently one of the de facto standards when performing LLM evaluation but it is an issue when the datasets that
you want to evaluate are supposed to be private.
