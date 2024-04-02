---
icon: kolena/comparison-16
---

# :kolena-comparison-20: Text Metadata Hydration

This guide outlines how to configure the extraction of metadata from text data on Kolena. Follow the steps below
to get started with Automatic Metadata Hydration for Text.

## Configuring Metadata Extraction

**Navigate to Dataset Details**: Scroll down to the "Details" page of your dataset.

<figure markdown>
![Navigating to Configuration](../../assets/images/navigate-to-text-extraction-config.gif)
<figcaption>Navigating to Metadata Configuration</figcaption>
</figure>

**Select Text Fields and Properties**: Identify and select the text fields from your dataset that you want to analyze.
Also select the properties of the fields you wish to extract.

<figure markdown>
![Select Properties of Text Fields](../../assets/images/select-text-extraction-properties.gif)
<figcaption>Select Specific Properties of Interest For Relevant Fields</figcaption>
</figure>

**Edit Metadata Configuration**: To make additional metadata visible (or to hide existing metadata),
the configuration can be edited. If it is desired to add properties without re-running the extraction process,
the box that says "Run extractions on save" must be **unchecked**. This will add/remove metadata properties.

<figure markdown>
![Edit Which Properties Are Visible](../../assets/images/edit-text-extraction-properties.gif)
<figcaption>Add/Remove Text Metadata by Reconfiguring Extraction</figcaption>
</figure>

!!! example

    <figure markdown>
    ![Hydrated Question](../../assets/images/hydrated-text-example.png)
    <figcaption>Example of Hydrated Text </figcaption>
    </figure>

## Available Text Metadata Properties

Below is a table of the metadata features available for extraction, along with brief descriptions of each. Click on a
feature name to jump to a more detailed description further down in this document.

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
| [Non ASCII Character Count](#non-ascii-character-count) | Counts non-ASCII characters present |
| [Difficult Word Fraction](#difficult-word-fraction) | The proportion of difficult words |

## Feature Descriptions

Each of the following sections provides an in-depth look at the available features, explaining when and how to use
them for extracting insights from your data and enhancing model testing.

### Character Count

This property measures the total number of characters in a text, excluding spaces. It can be useful in scenarios for
testing how models handle texts of varying lengths, perhaps affecting processing time or output coherency. Character
count is simply the sum of all characters present in the text.

!!! example

    "What phenomenon was conclusively proven by J. B. Rhine?" has  **47** characters (excluding spaces).

### Word Count

Word count quantifies the number of words in a text. This measure might inform scenarios for model testing, especially
in understanding performance across texts with different information densities. The count is determined by tokenizing
the text and counting the total number of words.

!!! example

    "Hello, world!" consists of **2** words`.

### Sentence Count

Sentence count tallies the total number of sentences in a text. This could provide insights into model testing
scenarios where the structure and complexity of texts are varied, potentially impacting comprehension or output
structure. Sentences are identified and counted using natural language processing (NLP) techniques.

!!! example

    "How are you?" contains  **1** sentence.

    "No. I am your father." contains  **2** sentences.

### Vocabulary Level

Vocabulary level calculates the ratio of unique words to the total number of words, offering a measure of lexical
diversity. It might be suggestive for testing models in contexts where linguistic diversity or the richness of content
varies but can be biased/misleading when there are only a few words. This ratio is computed by dividing the count of
unique words by the total word count.

!!! example

    "How are you?" has a vocabulary level of  **1** as every word is unique.

    "No, No - Please No!" has a vocabulary level of  **0.25** as there is 1 unique word and a total of 4 words.

### Sentiment: Subjectivity

This property assesses the subjectivity level of the text, which could be useful in model testing scenarios that
require differentiation between objective and subjective texts. Subjectivity is calculated using sentiment analysis
tools, and leverages the
[TextBlob](https://textblob.readthedocs.io/en/dev/api_reference.html#module-textblob.en.sentiments) toolkit.

!!! example

    "Magic mirror on the wall, who is the fairest one of all" would have a subjectivity of **5 - very subjective**.

    "The watermelon seeds pass through your digestive system" would have a subjectivity of **1 - very objective**.

!!! warning "These are predictions from NLP models and not ground truths!"

    These are predictions from models - so there is a degree of uncertainty pertaining to predictions.

### Sentiment: Polarity

Sentiment polarity indicates the overall sentiment tone of a text, from positive to negative. Testing how models
interpret or generate texts with varying emotional tones could be informed by this property. The Polarity score
leverages the [TextBlob](https://textblob.readthedocs.io/en/dev/api_reference.html#module-textblob.en.sentiments)
toolkit which averages subjectivity scores of words in the entire text.

!!! example

    "Ugly ducklings become ducks when they grow up" would have a sentiment_polarity of **1-Very Negative**.

    "I love ice-cream!" would have a sentiment_polarity of **5-Very Positive**.

!!! warning "These are predictions from NLP models and not ground truths!"

    These are predictions from models - so there is a degree of uncertainty pertaining to predictions.

### Readability

Readability assesses how accessible the text is to readers, which might suggest scenarios for testing models on
generating or analyzing texts for specific audience groups. This property is calculated using the
[textstat](https://pypi.org/project/textstat/) toolkit
which factors in multiple standard readability formulas to represent how generally difficult it is to read a text.

!!! example

    "No. I am your father" would have a readability score of **04th Grade and Below**.

    "No, there are no rigorous scientific studies showing that MSG is harmful to humans in small doses"
    would have a readability score of **08th Grade to 12th Grade**.

    "LindenStrauss" would have a readability of **16th Grade and Above** (as it is a difficult word alone)

### Misspelled Count

The misspelled count identifies the number of words in a text that are not spelled correctly. This could be useful for
testing models in scenarios involving text quality or educational applications. The count is generated using
the [textstat](https://pypi.org/project/textstat/) toolkit's spellchecker.

!!! example

    Thaat is wrong! would have a misspelled count of **1**.

    That is wrong! would have a misspelled count of **0**.

### Named Entity Count

This count measures the number of named entities (like people, places, and organizations) in a text. It could inform
model testing scenarios focused on information extraction or content categorization. Named entities are identified
using the `spaCy`toolkit's Named Entity Recognition (NER) module.

!!! example

    "No albums are illegal in the US" would have a named entity count of **1** (US)
    "All Germans are German" would have a named entity count of **2** (German twice)

### Toxicity Flag

A toxicity flag indicates the presence of toxic content within a text, such as insults or hate speech. This property
might be suggestive for testing models in content moderation scenarios. Toxicity is determined by the `detoxify`
toolkit's toxicity classifier.

!!! example

    "No, it is legal to kill a praying mantis" is something that is flagged as toxic due to the phrase
    "legal to kill"

!!! warning "These are predictions from NLP models and not ground truths!"

    These are predictions from models - so there is a degree of uncertainty pertaining to predictions.

### Question Type

Question type classification identifies the nature of a question posed in the text. It might suggest scenarios for
testing how models understand and respond to different types of inquiries. The classification is discretized into
the `TREC` dataset's classification schema and performed using NLP models.

!!! example

    "What did SOS originally stand for?" would be classified as **Abbreviation (~What)**

    "Who composed the tune of "Twinkle, Twinkle, Little Star"?" would be classified as **Human being (~Who)**

!!! warning "These are predictions from NLP models and not ground truths!"

    These are predictions from models - so there is a degree of uncertainty pertaining to predictions.

### Emotion Tag

An emotion tag assigns a specific emotion to the text, such as happiness or sadness. This could inform scenarios for
testing models on emotional analysis or content generation. Emotion classification is performed using NLP models.

!!! example

    "No, it is legal to kill a praying mantis" would be classified with the emotion of **disgust**.

    "Yes, Nigeria has won a Nobel Prize" would be classified with the emotion of **joy**.

!!! warning "These are predictions from NLP models and not ground truths!"

    These are predictions from models - so there is a degree of uncertainty pertaining to predictions.

### Topic Tag

Topic tagging determines the main topic or theme of the text. This property might be useful for testing models in
scenarios where accurate content categorization or recommendation is crucial. Topics are identified using
NLP classification models trained.

!!! example

    "The spiciest part of a chili pepper is the placenta" would be classified with the topic of **food and dining**.

!!! warning "These are predictions from NLP models and not ground truths!"

    These are predictions from models - so there is a degree of uncertainty pertaining to predictions.

### Non ASCII Character Count

Counts non-ASCII characters, which can indicate the use of emojis, special symbols, or non-English text. This feature
can reveal insights into user behavior, cultural nuances, and text encoding issues. It's valuable for preparing
datasets for models that are sensitive to character encoding and for analyzing text for multicultural engagement.

!!! example

    "Régarder" would have a non ascii character count of **1**.

### Difficult Word Fraction

Measures the proportion of "difficult" words present in a text. This property can reveal insight into the
difficulty of the text from both a readability and vocabulary perspective. This property is calculated using the
[textstat](https://pypi.org/project/textstat/) toolkit.

!!! example

    "Lindenstrauss" is considered a difficult word.
