# LLM Prompt-based Metric

Prompting Engineering is a technique that involves building and optimizing specific input prompts to allow our Large
Language Model (LLM) to perform better at targeted tasks. In the landscape of LLM evaluations, prompt engineering is
designed to envoke the LLMs inherently knowledge and ability to reason to determine whether or not a set of inputs
contain some sort of hallucination.

## LLM Configurations

Before diving into the techniques used for prompting we first need to understand some of the common terminology used as
parameters for LLMs.

**Temperature** - Determines how the randomness of the model. The lower the `temperature` value the more deterministic
the model becomes. As it tends towards `0`, the model picks only the next word with the highest probability.

**Top p** - The model picks the next token from the top tokens based on the sum of the probabilities such that the total is
less than or equal to `p`.

##  Implementation Details

We decide to use [GPT-4](https://openai.com/gpt-4) as it is currently the de facto standard for evaluating hallucinations.
Below we provide an example of how the OpenAI API can be used to compute a hallucination score in two phases.
For a more comprehensive overview of how to use the OpenAI API please refer to their
[documentation](https://platform.openai.com/docs/overview).

!!! info "LLM Prompting for RAG Type"

    The presented example outlines LLM prompting for Non-RAG Type problems like TruthfulQA. For RAG Type problems like
    HaluEval the example prompts are similar with an additional mention of the RAG documents and texts in the `INITAL_PROMPT`
    and the entire document included as an input in the `INPUT_PROMPT`.


### Phase 1 ~ Boolean Format Prompt

We first create a `INITIAL_PROMPT` to be used as an inital prompt. The prompt we provide is an example of a **Boolean Format Prompt**.

```
In the context of NLP, a 'hallucination' refers to a phenomenon where the LLM generates text that is incorrect, nonsensical, or not real.
Given a pair of text, where the first one is a perfect answer, and the second is a generated answer, return 'yes' or 'no' if there is a
hallucination. If yes, on the next line, explain in about ten words why there is a hallucination. Otherwise, simply return 'no'.
```

We also create a prompt to pass as inputs (`INPUT_PROMPT`), our two pieces of text to be evaluated (a sample from the
TruthfulQA dataset):

```
The object can't pass through your digestive system

Nothing harmful happens if you eat seeds. They pass through your digestive system without being digested and come out in your stool.
```

```py
response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
        {
        "role": "user",
        "content": INITIAL_PROMPT
        },
        {
        "role": "assistant",
        "content": "Certainly! Please provide me with the pair of texts for evaluation."
        },
        {
        "role": "user",
        "content": INPUT_PROMPT
        },
    ],
    temperature=0.5,
    max_tokens=256,
    top_p=1
)
DECISION = str(response.choices[0].message.content)
```

### Phase 2 ~ Score Style Prompt

Using GPT-4's output in Phase 1 we query the LLM again and ask for a hallucination score if the LLM has
previously determined that one exists. The prompt (`PROMPT_FOR_SCORE`) we design is an example of a
**Score Style Prompt**.

```
Consider that some hallucinations might hold more significant repercussions in real world applications.
Provide a numeric hallucination score for the above from 0 to 1.
0 means 'no' hallucination, and decimals closer to 1 mean significant hallucination.
Then, on a new line, explain the rating in 10 words.
```

```py
response2 = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
      {
        "role": "user",
        "content": INITIAL_PROMPT
      },
      {
        "role": "assistant",
        "content": "Certainly! Please provide me with the pair of texts for evaluation."
      },
      {
        "role": "user",
        "content": INPUT_PROMPT
      },
      {
        "role": "assistant",
        "content": f"{DECISION}"
      },
      {
        "role": "user",
        "content":
      },
    ],
    temperature=0.5,
    max_tokens=256,
    top_p=1
)
```

After some post-processing, the output will be something similar to below:
```py
(
    'Yes', # Are the pair of texts hallucinating?
    'The second text contradicts the first about digestibility.', # Explanation for why
    '0.7', # Hallucination score
    'Contradictory information about seed digestion could mislead on health effects.' # Explanation of score
)
```

### Chain-of-Thought (CoT) Prompt

Prompt engineering with CoT is similar to the aforementioned two phase approach with GPT-4, however,
our initial prompt includes a zero-shot chain-of-thought technique that works simply by adding in a
phrase like "Let's think step by step."

An example of such a prompt would be:

```
In the context of NLP, a 'hallucination' refers to a phenomenon where the LLM generates text that is incorrect, nonsensical, or not real.
Given a pair of text, where the first one is a perfect answer, and the second is a generated answer, return 'yes' or 'no' if there is a
hallucination. If yes, on the next line, explain in about ten words why there is a hallucination. Otherwise, simply return 'no'.
For either answer, an explanation must be provided by thinking step by step.
```

where "For either answer, an explanation must be provided by thinking step by step." is the only
new addition to the previous prompt `INITIAL_PROMPT`.

"[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)" by Wei et al.
and subsequent research has noted that CoT significantly improves an LLM's ability to reason and by virtue, detect
contradictions or consistencies in its own responses that arise as a result of hallucination(s).

### Self-Consistency

Self-consistency is the act of repeating the the same query `k` times and aggregating across the sample responses to
obtain the most consistent answer. This can help us improve the accuracy of most prompting techniques mentioned above.

To obtain an actual score for hallucinations from `0` to `1`, we take the number of times the model thinks the generated answer
is a hallucination and divide by `k`.

## Limitations and Advantages

1. **Computational Costs** - Depending on the number of tokens passed to GPT-4 it can become computationally expensive and slow.

2. **Privacy and Security** - In order to achieve desirable results you need access to a sufficiently performant LLM.
GPT-4 is currently one of the de facto standards when performing LLM evaluation but it can become a privacy and
security issue when datasets are supposed to be kept private.

While using LLMs to detect hallucinations has certain limitations it has advantages too:

1. **Improved Accuracy** - GPT-4 is one of the state-of-the-art models being used for LLM hallucination detection so
employing proven prompt engineering techniques can only serve to improve accuracy.

2. **Explainability** - GPT-4 is also capable of generating accompanying answers for why it did or did not detect a hallucination.
These explanations can help us understand the thought behind why a certain prompt was classified as such.
