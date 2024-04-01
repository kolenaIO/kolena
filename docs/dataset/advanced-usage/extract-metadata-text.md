# Text Data Analysis Feature Configuration Guide

This guide outlines how to configure the extraction of metadata from text data within our platform. Follow these steps
to tailor the analysis to your specific needs, ensuring you extract the most relevant information from your datasets.

## Configuring Metadata Extraction

To begin, navigate to the dataset's "Details" page to configure the metadata you wish to extract. This involves
selecting which text fields you want to analyze and specifying the type of metadata to extract for each field.

1. **Navigate to Dataset Details**: Scroll down to the "Details" page of your dataset.

2. **Select Text Fields**: Identify and select the text fields from your dataset that you want to analyze.

3. **Configure Metadata Extraction**: For each selected text field, choose the metadata types you wish to extract.

## Available Features Quick Reference Table

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

## Feature Descriptions

Each of the following sections provides an in-depth look at the available features, explaining when and how to use
them for extracting insights from your data and enhancing model testing.

### Character Count

This property measures the total number of characters in a text, excluding spaces. It can be useful in scenarios for
testing how models handle texts of varying lengths, perhaps affecting processing time or output coherency. Character
count is simply the sum of all characters present in the text.

### Word Count

Word count quantifies the number of words in a text. This measure might inform scenarios for model testing, especially
in understanding performance across texts with different information densities. The count is determined by tokenizing
the text and counting the total number of words.

### Sentence Count

Sentence count tallies the total number of sentences in a text. This could provide insights into model testing
scenarios where the structure and complexity of texts are varied, potentially impacting comprehension or output
structure. Sentences are identified and counted using natural language processing (NLP) techniques.

### Vocabulary Level

Vocabulary level calculates the ratio of unique words to the total number of words, offering a measure of lexical
diversity. It might be suggestive for testing models in contexts where linguistic diversity or the richness of content
varies but can be biased/misleading when there are only a few words. This ratio is computed by dividing the count of
unique words by the total word count.

### Sentiment: Subjectivity

This property assesses the subjectivity level of the text, which could be useful in model testing scenarios that
require differentiation between objective and subjective texts. Subjectivity is calculated using sentiment analysis
tools, and leverages the `Textblob` toolkit which averages subjectivity scores of words in the entire text.

### Sentiment: Polarity

Sentiment polarity indicates the overall sentiment tone of a text, from positive to negative. Testing how models
interpret or generate texts with varying emotional tones could be informed by this property. The Polarity score
leverages the `Textblob` toolkit which averages subjectivity scores of words in the entire text.

### Readability

Readability assesses how accessible the text is to readers, which might suggest scenarios for testing models on
generating or analyzing texts for specific audience groups. This property is calculated using the `textstat` toolkit
which factors in multiple standard readability formulas to represent how generally difficult it is to read a text.

### Misspelled Count

The misspelled count identifies the number of words in a text that are not spelled correctly. This could be useful for
testing models in scenarios involving text quality or educational applications. The count is generated using
spell-check libraries against a standard dictionary.

### Named Entity Count

This count measures the number of named entities (like people, places, and organizations) in a text. It could inform
model testing scenarios focused on information extraction or content categorization. Named entities are identified
using NLP entity recognition techniques.

### Toxicity Flag

A toxicity flag indicates the presence of toxic content within a text, such as insults or hate speech. This property
might be suggestive for testing models in content moderation scenarios. Toxicity is usually determined using models
trained on datasets labeled for various forms of inappropriate content.

### Question Type

Question type classification identifies the nature of a question posed in the text. It might suggest scenarios for
testing how models understand and respond to different types of inquiries. The classification is typically achieved
through NLP models trained on question-answer datasets.

### Emotion Tag

An emotion tag assigns a specific emotion to the text, such as happiness or sadness. This could inform scenarios for
testing models on emotional analysis or content generation. Emotion classification is usually performed using
sentiment analysis models trained on emotionally annotated texts.

### Topic Tag

Topic tagging determines the main topic or theme of the text. This property might be useful for testing models in
scenarios where accurate content categorization or recommendation is crucial. Topics are typically identified using
NLP classification models trained on a variety of topic-labeled data.

### [Additional Feature Sections Follow the Same Format]

...

### Non ASCII Character Count

Counts non-ASCII characters, which can indicate the use of emojis, special symbols, or non-English text. This feature
can reveal insights into user behavior, cultural nuances, and text encoding issues. It's valuable for preparing
datasets for models that are sensitive to character encoding and for analyzing text for multicultural engagement.

Placeholder for image or GIF on Non ASCII Character Count: `![Non ASCII Character Count](url-to-image-or-gif)`

...

Remember to replace `url-to-image-or-gif` with the actual URLs to your images or GIFs. This documentation will guide
users through configuring the text data analysis features, ensuring they can effectively extract and utilize the
metadata for their specific needs.
