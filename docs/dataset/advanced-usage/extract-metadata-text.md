---
icon: kolena/properties-16
---

# :kolena-properties-20: Text Metadata Extraction

This guide outlines how to configure the extraction of metadata from text data on Kolena. Follow the steps below
to get started with Automatic Metadata Extraction for Text.

## Configuring Metadata Extraction

??? "1. Navigate to Dataset Details"
    Scroll down to the "Details" page of your dataset.

    <figure markdown>
    ![Navigating to Configuration](../../assets/images/navigate-to-text-extraction-config.gif)
    <figcaption>Navigating to Metadata Configuration</figcaption>
    </figure>

??? "2. Select Text Fields and Properties"
    Identify and select the text fields from your dataset that you want to analyze.
    Also select the properties of the fields you wish to extract.

    In the examble below we extract properties from the `best_answer` and `question` fields. For the `best_answer` field,
    we display `word_count` and `topic_tag`, whereas for the `question` field we display `word_count`, `readability` and
    `question_type`.

    <figure markdown>
    ![Select Properties of Text Fields](../../assets/images/select-text-extraction-properties.gif)
    <figcaption>Select Specific Properties of Interest For Relevant Fields</figcaption>
    </figure>

??? "3. Edit Metadata Configuration"
    To make additional metadata visible (or to hide existing metadata),
    the configuration can be edited. If it is desired to add properties without re-running the extraction process,
    the box that says "Run extractions on save" must be **unchecked** otherwise the pipeline is re-run.

    This will add/remove metadata properties. The example below shows how to add the `character_count` property
    to the `best_answer` field without re-running the extraction pipeline.

    <figure markdown>
    ![Edit Which Properties Are Visible](../../assets/images/edit-text-extraction-properties.gif)
    <figcaption>Example of adding `character_count` to the list of extracted properties</figcaption>
    </figure>

!!! example

    <figure markdown>
    ![Hydrated Question](../../assets/images/hydrated-text-example.png)
    <figcaption>Example of Hydrated Text </figcaption>
    </figure>

## Available Text Metadata Properties

The following properties are available for automatic text metadata extraction:

| Feature Name                  | Brief Description                              |
|-------------------------------|------------------------------------------------|
| [Character Count](#character-count) | Counts all characters, excluding spaces |
| [Word Count](#word-count) | Measures the total number of words |
| [Sentence Count](#sentence-count) | Tallies the sentences in the text |
| [Vocabulary Level](#vocabulary-level) | Ratio of unique words to total words |
| [Sentiment: Subjectivity](#sentiment-subjectivity) | Subjectivity score of the text |
| [Sentiment: Polarity](#sentiment-polarity) | Polarity score indicating sentiment tone |
| [Readability](#readability) | Assessment of text readability level |
| [Misspelled Count](#misspelled-count) | Count of misspelled words |
| [Named Entity Count](#named-entity-count) | Number of named entities in the text |
| [Toxicity Flag](#toxicity-flag) | Flags potentially toxic content |
| [Question Type](#question-type) | Identifies the type of question posed |
| [Emotion Tag](#emotion-tag) | Classifies the text's associated emotion |
| [Topic Tag](#topic-tag) | Determines the overarching topic |
| [Non-ASCII Character Count](#non-ascii-character-count) | Counts non-ASCII characters present |
| [Difficult Word Fraction](#difficult-word-fraction) | The proportion of difficult words |

## Feature Descriptions

### Character Count

**Character count** measures the total number of characters in a text, excluding spaces. It can be useful in scenarios
for testing how models handle texts of varying lengths, perhaps affecting processing time or output coherency.
Character count is simply the sum of all characters present in the text.

!!! example

    "What phenomenon was conclusively proven by J. B. Rhine?" has  **47** characters (excluding spaces).

### Word Count

**Word count** quantifies the number of words in a text. This measure might inform scenarios for model testing,
especially in understanding performance across texts with different information densities. The count is determined by
tokenizing the text using the [nltk toolkit](https://www.nltk.org/) and counting the total number of words.

!!! example

    "Hello, world!" consists of **2** words.

### Sentence Count

**Sentence count** tallies the total number of sentences in a text. This could provide insights into model testing
scenarios where the structure and complexity of texts are varied, potentially impacting comprehension or output
structure. Sentences are identified and counted using the [nltk toolkit](https://www.nltk.org/)'s sentence tokenizer.

!!! example

    "How are you?" contains  **1** sentence.

    "No. I am your father." contains  **2** sentences.

### Vocabulary Level

**Vocabulary level**  calculates the ratio of unique words to the total number of words, offering a measure of lexical
diversity. It might be suggestive for testing models in contexts where linguistic diversity or the richness of content
varies but can be biased/misleading when there are only a few words. This ratio is computed by dividing the count of
unique words by the total word count.

!!! example

    "How are you?" has a vocabulary level of  **1** as every word is unique.

    "No, No - Please No!" has a vocabulary level of  **0.25** as there is 1 unique word and a total of 4 words.

### Sentiment: Subjectivity

**Sentiment subjectivity** assesses the subjectivity level of the text, which could be useful in model testing
scenarios that require differentiation between objective and subjective texts. Subjectivity is calculated using
the
[TextBlob](https://textblob.readthedocs.io/en/dev/api_reference.html#module-textblob.en.sentiments) toolkit.

??? "Subjectivity Levels"
    There are 5 possible levels of subjectivity supported:

    1. very objective
    2. mildly objective
    3. neutral
    4. mildly subjective
    5. very subjective

!!! example

    "Magic mirror on the wall, who is the fairest one of all" would have a subjectivity of **5 - very subjective**.

    "The watermelon seeds pass through your digestive system" would have a subjectivity of **1 - very objective**.

!!! info "These are predictions from NLP models and not ground truths!"

    These are predictions from models - so there is a degree of uncertainty pertaining to predictions.

### Sentiment: Polarity

**Sentiment polarity** indicates the overall sentiment tone of a text, from positive to negative. Testing how models
interpret or generate texts with varying emotional tones could be informed by this property. The polarity score
is calculated using the
[TextBlob](https://textblob.readthedocs.io/en/dev/api_reference.html#module-textblob.en.sentiments)
toolkit.

??? "Polarity Levels"
    There are 5 possible levels of polarity supported:

    1. very negative
    2. mildly negative
    3. neutral
    4. mildly positive
    5. very positive

!!! example

    "Ugly ducklings become ducks when they grow up" would have a sentiment_polarity of **1-very negative**.

    "I love ice-cream!" would have a sentiment_polarity of **5-very positive**.

!!! info "These are predictions from NLP models and not ground truths!"

    These are predictions from models - so there is a degree of uncertainty pertaining to predictions.

### Readability

**Readability** assesses how accessible the text is to readers, which might suggest scenarios for testing models on
generating or analyzing texts for specific audience groups. This property is calculated using the
[textstat](https://pypi.org/project/textstat/) toolkit
which factors in multiple standard readability formulas to represent how generally difficult it is to read a text.

??? "Readability Levels"
    There are 5 possible levels of readability supported:

    1. 04th Grade and Below
    2. 04th Grade to 08th Grade
    3. 08th Grade to 12th Grade
    4. 12th Grade to 16th Grade
    5. 16th Grade and Above

!!! example

    "No. I am your father" would have a readability score of **04th Grade and Below**.

    "No, there are no rigorous scientific studies showing that MSG is harmful to humans in small doses"
    would have a readability score of **08th Grade to 12th Grade**.

    "LindenStrauss" would have a readability of **16th Grade and Above** (as it is a difficult word alone)

### Misspelled Count

**Misspelled count** identifies the number of words in a text that are not spelled correctly. This could be useful for
testing models in scenarios involving text quality or educational applications. The count is generated using
the [textstat](https://pypi.org/project/textstat/) toolkit's spellchecker. This property can often be a proxy to
unrecognized named entities as well.

!!! example

    "Thaat is wrong!" would have a misspelled count of **1**.

    "That is wrong!" would have a misspelled count of **0**.

### Named Entity Count

**Named entity count** measures the number of named entities (like people, places, and organizations) in a text.
It could inform model testing scenarios focused on information extraction or content categorization.
Named entities are identified using
the [spaCy](https://spacy.io/) toolkit's Named Entity Recognition (NER) module.

!!! example

    "No albums are illegal in the US" would have a named entity count of **1** (US)

    "All Germans are German" would have a named entity count of **2** (German twice)

### Toxicity Flag

A **Toxicity flag** indicates the presence of toxic content within a text, such as insults or hate speech.
This property might be suggestive for testing models in content moderation scenarios. Toxicity is determined by the
[detoxify](https://pypi.org/project/detoxify/) toolkit's toxicity classifier.

!!! example

    "No, it is legal to kill a praying mantis" is something that is flagged as toxic due to the phrase
    "legal to kill"

!!! info "These are predictions from NLP models and not ground truths!"

    These are predictions from models - so there is a degree of uncertainty pertaining to predictions.

### Question Type

**Question type** classification identifies the nature of a question posed in the text. It might suggest scenarios for
testing how models understand and respond to different types of inquiries. The classification is discretized into
the [TREC](https://huggingface.co/datasets/trec) dataset's
classification schema and performed using an [NLP classification model](https://huggingface.co/datasets/trec).

??? "Question Types"
    There are 6 possible question types supported:

    1. Abbreviation (~What)
    2. Entity (~What)
    3. Description (~Describe)
    4. Human being (~Who)
    5. Location (~Where)
    6. Numeric (~How Much)

!!! example

    "What did SOS originally stand for?" would be classified as **Abbreviation (~What)**

    "Who composed the tune of "Twinkle, Twinkle, Little Star"?" would be classified as **Human being (~Who)**

!!! info "These are predictions from NLP models and not ground truths!"

    These are predictions from models - so there is a degree of uncertainty pertaining to predictions.

### Emotion Tag

An **Emotion tag** assigns a specific emotion to the text, such as happiness or sadness. This could give insight into
how models on texts with different emotional undertones. Emotion classification is performed using an
[NLP classification model](https://huggingface.co/michellejieli/emotion_text_classifier).

??? Emotions
    The 7 following emotions are supported:

    1. Anger
    2. Disgust
    3. Fear
    4. Joy
    5. Neutral
    6. Sadness
    7. Surprise

!!! example

    "No, it is legal to kill a praying mantis" would be classified with the emotion of **disgust**.

    "Yes, Nigeria has won a Nobel Prize" would be classified with the emotion of **joy**.

!!! info "These are predictions from NLP models and not ground truths!"

    These are predictions from models - so there is a degree of uncertainty pertaining to predictions.

### Topic Tag

**Topic tagging** determines the main topic or theme of the text. This property might be useful to gauge model
performance with relation to different topics pertaining to content. Topics are identified using inferences
from an [NLP classification model](https://huggingface.co/cardiffnlp/tweet-topic-21-multi).

??? Topics
    The following topics are supported:

    1. business & entrepreneurs
    2. celebrity & pop_culture
    3. diaries & daily life
    4. family
    5. fashion & style
    6. film_tv & video
    7. fitness & health
    8. food & dining
    9. gaming
    10. learning & educational
    11. music
    12. news & social_concern
    13. other hobbies
    14. relationships
    15. science & technology
    16. sports
    17. travel & adventure
    18. youth & student life
    19. arts & culture

!!! example

    "The spiciest part of a chili pepper is the placenta" would be classified with the topic of **food and dining**.

!!! info "These are predictions from NLP models and not ground truths!"

    These are predictions from models - so there is a degree of uncertainty pertaining to predictions.

### Non-ASCII Character Count

Counts **non-ASCII characters**, which can indicate the use of emojis, special symbols, or non-English text. This feature
could be potentially helpful in ascertaining how models deal with non-ascii characters, i.e when there are multiple
languages in the same text, etc.

!!! example

    "Régarder" would have a non ascii character count of **1**.

### Difficult Word Fraction

**Difficult word fraction** measures the proportion of "difficult" words present in a text.
This property can reveal insight into the
difficulty of the text from both a readability and vocabulary perspective. This property is calculated using the
[textstat](https://pypi.org/project/textstat/) toolkit.

!!! example

    "Lindenstrauss" is considered a difficult word.
