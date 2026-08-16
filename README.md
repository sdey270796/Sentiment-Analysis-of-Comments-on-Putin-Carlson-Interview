# Sentiment Analysis of Putin--Carlson Interview Comments

An exploratory NLP project that separates likely bot-generated comments
from genuine user comments and compares their sentiment, engagement, and
behavioral patterns.

## 📌 Project Overview

This project analyzes a dataset of **100,000 YouTube comments**
associated with the Vladimir Putin--Tucker Carlson interview.

Rather than performing sentiment analysis on the complete dataset
without qualification, the notebook first investigates whether the
comments contain strong signals of automated or duplicated activity.
Comments that occur more than once are treated as **likely bot-generated
comments**, while comments occurring only once are treated as **likely
genuine user comments**.

The two groups are then analyzed separately using **NLTK's VADER
sentiment analyzer**, with additional investigation of likes, replies,
publication timing, repetition patterns, and the textual characteristics
of bot comments.

The project therefore combines:

-   Exploratory Data Analysis
-   Duplicate/repetition analysis
-   Heuristic bot-comment detection
-   Natural Language Processing
-   Sentiment analysis
-   Engagement analysis
-   Behavioral pattern analysis
-   Visualization and interpretation

------------------------------------------------------------------------

## 🎯 Objectives

The notebook aims to:

1.  Examine the structure and quality of the comment dataset.
2.  Identify repeated comments and investigate whether they exhibit
    bot-like behavior.
3.  Separate likely bot-generated comments from likely genuine comments.
4.  Compare engagement patterns between the two groups.
5.  Perform sentiment analysis independently on bot and genuine
    comments.
6.  Examine the sentiment distribution of both groups.
7.  Investigate whether bot comments exhibit common behavioral or
    engagement patterns.
8.  Explore whether the detected bot activity could distort
    interpretations of public sentiment.

------------------------------------------------------------------------

## 🗂️ Dataset

The dataset is loaded from:

``` text
putin_tucker.csv
```

The notebook obtains it from the associated GitHub repository and loads
it into a Pandas DataFrame.

### Dataset Size

The notebook reports:

-   **100,000 rows**
-   **5 columns**
-   **No missing values**

### Columns

  -----------------------------------------------------------------------
  Column                              Description
  ----------------------------------- -----------------------------------
  `Comment`                           Textual content of the YouTube
                                      comment

  `Anonymized Author`                 SHA-256-hashed representation of
                                      the author's username

  `Published At`                      Timestamp at which the comment was
                                      published

  `Likes`                             Number of likes received by the
                                      comment

  `Reply Count`                       Number of replies received by the
                                      comment
  -----------------------------------------------------------------------

The notebook describes the dataset as ethically mined, with author
identifiers anonymized.

------------------------------------------------------------------------

## 🔍 Initial Data Exploration

The notebook begins with standard data-quality checks:

-   Dataset shape
-   Descriptive statistics
-   Data types
-   Missing-value inspection

The numerical variables show highly skewed engagement distributions.

### Engagement Statistics

  Metric       Likes   Reply Count
  --------- -------- -------------
  Mean         52.23          2.39
  Median           2             0
  Maximum     79,514           750

The large difference between the mean and median indicates that
engagement is strongly influenced by a relatively small number of highly
engaged comments.

------------------------------------------------------------------------

## 🤖 Bot-Comment Detection

### Core Heuristic

The project uses a simple but highly interpretable heuristic:

> **Any comment appearing more than once in the dataset is classified as
> a likely bot comment.**

This is implemented by counting how many times each exact comment text
occurs and creating the `DEV__IsBotComment` indicator.

``` python
unique_comment_dist = df.groupby('Comment')['Anonymized Author'].count()

df['DEV__IsBotComment'] = df['Comment'].isin(
    unique_comment_dist[unique_comment_dist > 1].index
)
```

### Result

The notebook identifies:

  Group                          Comments   Percentage
  ------------------------- ------------- ------------
  Likely bot comments          **99,702**   **99.70%**
  Likely genuine comments         **298**    **0.30%**
  **Total**                   **100,000**     **100%**

This is an extremely strong imbalance.

### Important Interpretation

This classification is a **heuristic**, not a definitive bot-detection
model.

Repeated text is treated as evidence of automated activity because the
notebook finds additional patterns supporting this interpretation,
including unusually large repetition clusters and highly consistent
engagement behavior.

However, repeated comments alone cannot mathematically prove that every
repeated comment was generated by a bot.

------------------------------------------------------------------------

## 📊 Repetition Analysis

The notebook examines the number of times identical comment texts occur.

The analysis reveals distinctive repetition clusters, including very
large repeated-comment groups. In particular, the notebook highlights
clusters around **997 and 1,994 repetitions**.

These unusually regular repetition counts are interpreted as strong
indicators of automated or coordinated comment activity.

The notebook also observes that:

-   Most comments occur only once.
-   A small number of comment texts occur repeatedly.
-   Some repeated comments appear in large, highly regular clusters.
-   The repetition structure is substantially different from what would
    normally be expected from independent organic comments.

A scatter plot is used to visualize the distribution of repetition
counts for unique comment texts.

------------------------------------------------------------------------

## ❤️ Engagement Analysis

The project compares the number of likes received by likely genuine and
bot comments.

Because likes span several orders of magnitude, the notebook uses:

-   Logarithmic bucketing
-   Base-2 transformation
-   An offset to avoid taking the logarithm of zero

### Likes: Genuine vs Bot Comments

  Statistic           Likely Genuine   Likely Bot
  ----------------- ---------------- ------------
  Count                          298       99,702
  Mean likes                  766.22        50.09
  Median likes                     8            2
  75th percentile               52.5           11
  90th percentile              477.2          214
  99th percentile          19,898.09          471
  Maximum                     79,514          542

### Key Observation

Although likely genuine comments are vastly fewer, they exhibit much
larger variation in engagement.

The notebook specifically highlights that:

> The 99th percentile of likely genuine comments reaches approximately
> **19,898 likes**, compared with **471 likes** for likely bot comments.

The engagement distributions are visualized using side-by-side
histograms.

The notebook interprets the comparatively structured bot engagement
distribution as evidence that engagement associated with repeated
comments may have been controlled or predetermined.

------------------------------------------------------------------------

## 🧹 Separating the Two Groups

After classification, the dataset is divided into two working
DataFrames:

``` python
df_gen
```

for likely genuine comments, and:

``` python
df_bot
```

for likely bot comments.

For bot comments, the notebook further aggregates repeated comments so
that each unique repeated comment can be examined along with its:

-   Average likes
-   Average reply count
-   Number of repetitions

This allows behavioral patterns to be studied at the level of unique
repeated comment text rather than treating every duplicate row as
independent evidence.

------------------------------------------------------------------------

# 🧠 Sentiment Analysis

The sentiment-analysis stage uses **NLTK**.

The notebook uses:

-   `WordNetLemmatizer`
-   English stopword removal
-   NLTK tokenization
-   `SentimentIntensityAnalyzer` / VADER

### Text Preprocessing

Each comment undergoes:

1.  Lowercasing
2.  Tokenization
3.  Stopword removal
4.  Lemmatization
5.  Reconstruction into processed text

The processed text is then passed to VADER.

### Sentiment Components

For each comment, VADER produces:

-   `neg` --- negative sentiment score
-   `neu` --- neutral sentiment score
-   `pos` --- positive sentiment score
-   `compound` --- normalized overall sentiment score

The compound score ranges from approximately:

``` text
-1  ← Negative
 0  ← Neutral
+1  ← Positive
```

------------------------------------------------------------------------

## 📈 Sentiment Results

### Likely Genuine Comments

For the 298 likely genuine comments, the notebook reports:

  Metric             Mean
  ---------- ------------
  Negative         0.0747
  Neutral          0.6435
  Positive         0.2819
  Compound     **0.2933**

The compound scores range from **-0.9428 to 0.9903**.

This indicates a relatively broad sentiment spectrum, including strongly
negative and strongly positive comments.

### Likely Bot Comments

After collapsing repeated bot comments to unique comment texts, the
notebook analyzes **103 unique bot-comment texts**.

  Metric             Mean
  ---------- ------------
  Negative         0.0666
  Neutral          0.6914
  Positive         0.2323
  Compound     **0.2577**

The compound scores range from **-0.6416 to 0.9723**.

------------------------------------------------------------------------

## 📊 Sentiment Distribution

The notebook visualizes the compound sentiment distributions using
histograms for the two groups.

### Likely Genuine Comments

The distribution:

-   Has a strong concentration around neutral sentiment.
-   Contains a wider spread across negative and positive values.
-   Includes more extreme negative sentiment than the bot-comment group.

### Likely Bot Comments

The bot-comment distribution:

-   Is also concentrated around neutral sentiment.
-   Shows a comparatively narrower range.
-   Contains fewer strongly negative observations.
-   Extends into positive sentiment.

The notebook therefore interprets bot comments as being comparatively
more controlled toward neutral/positive sentiment, although the
sentiment analyzer itself is not sufficient to establish intent.

------------------------------------------------------------------------

# 🔬 Detailed Bot-Comment Analysis

The notebook goes beyond simple sentiment scoring and manually examines
the repeated comments.

Several patterns are highlighted.

### 1. Repeated Viewpoints

Many repeated comments appear to support particular viewpoints
concerning Russia and the interview.

The notebook notes that a substantial number appear supportive of
Russia, while opposing bot comments are also present.

### 2. Engagement-Bait Patterns

Some repeated comments appear to be designed primarily to attract
interaction rather than provide substantive commentary.

### 3. Multilingual Comments

The notebook identifies repeated comments in multiple languages,
sometimes appearing as isolated examples for individual languages.

This is interpreted as potentially simulating broader international
participation.

### 4. Synchronized Publication

Repeated comments frequently show extremely consistent publication and
engagement characteristics.

The notebook investigates exceptions by checking whether repeated
comments have:

-   Multiple publication timestamps
-   Multiple like counts
-   Multiple reply counts

This provides an additional behavioral check rather than relying solely
on text duplication.

------------------------------------------------------------------------

## ⏱️ Publication and Engagement Consistency

One of the more striking observations in the notebook is that many
repeated bot comments appear to have been published simultaneously and
to have identical engagement values.

For repeated comments, the notebook aggregates:

``` text
Likes → mean / count
Reply Count → mean / count
Repetition count
```

The analysis suggests that the engagement values associated with many
repeated comments may have been predetermined or systematically
controlled.

The notebook further observes that negative bot comments generally
appear to have lower likes and replies than positive bot comments.

This pattern is presented as a potential indicator of engineered
engagement rather than natural audience behavior.

------------------------------------------------------------------------

## 📊 Visualizations Included

The notebook contains multiple exploratory and analytical
visualizations, including:

### Comment Repetition

-   Distribution of repetition counts for unique comment texts
-   Scatter plot of unique comments vs repetition count

### Engagement

-   Likely genuine-comment like distribution
-   Likely bot-comment like distribution
-   Log-scaled engagement histograms

### Sentiment

-   Likely genuine-comment sentiment distribution
-   Likely bot-comment sentiment distribution

These visualizations are used not merely for presentation, but to
identify structural differences between the two groups.

------------------------------------------------------------------------

## 🛠️ Technologies Used

### Programming

-   Python
-   Jupyter Notebook / Google Colab

### Data Processing

-   Pandas
-   NumPy

### Natural Language Processing

-   NLTK
-   VADER Sentiment Analysis
-   WordNet Lemmatization
-   Tokenization
-   Stopword removal

### Visualization

-   Matplotlib
-   Seaborn

### Data Source

-   YouTube comment dataset
-   CSV format

------------------------------------------------------------------------

## 📁 Repository Structure

The repository can be organized as:

``` text
.
├── README.md
├── Sentiment_Analysis_of_Putin_Carlson_Interview_Comments.ipynb
├── putin_tucker.csv
└── ...
```

The notebook contains the complete analytical workflow, while the CSV
file contains the underlying comment dataset.

------------------------------------------------------------------------

## 🚀 How to Run

### 1. Clone the repository

``` bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install dependencies

``` bash
pip install numpy pandas matplotlib seaborn nltk
```

### 3. Open the notebook

Launch:

``` text
Sentiment_Analysis_of_Putin_Carlson_Interview_Comments.ipynb
```

The notebook downloads the required NLTK resources, including:

``` text
punkt
stopwords
wordnet
vader_lexicon
```

### 4. Load the dataset

The notebook expects:

``` text
putin_tucker.csv
```

to be available through the repository path used in the notebook.

### 5. Execute the notebook

Run the cells sequentially to reproduce:

1.  Dataset loading
2.  Data-quality checks
3.  Repetition analysis
4.  Bot/genuine segregation
5.  Engagement analysis
6.  Text preprocessing
7.  VADER sentiment analysis
8.  Bot-comment aggregation
9.  Sentiment visualization
10. Detailed bot-comment analysis
11. Final conclusions

------------------------------------------------------------------------

## ⚠️ Limitations and Important Caveats

### Heuristic Bot Detection

The project does **not** train a machine-learning classifier to detect
bots.

Instead, it uses exact comment repetition as the primary classification
rule.

Therefore:

``` text
Repeated comment ≠ mathematically proven bot
```

The classification becomes more persuasive because repeated comments
also exhibit other unusual behavioral patterns, but it should still be
regarded as heuristic.

### Small Genuine-Comment Sample

Only **298 of the 100,000 comments** are classified as likely genuine
under the notebook's rule.

Consequently, conclusions about genuine-user sentiment are based on a
relatively small sample.

### Sentiment Model Limitations

VADER is a lexicon/rule-based sentiment analyzer. It may fail to
understand:

-   Sarcasm
-   Political context
-   Irony
-   Domain-specific language
-   Multilingual text
-   Complex rhetorical statements

The notebook itself notes that sentiment scores do not always perfectly
correspond to the actual meaning of individual comments.

### Sentiment Is Not Intent

A positive sentiment score does not establish political support,
propaganda intent, or coordinated influence.

The behavioral evidence can suggest unusual activity, but determining
intent would require additional investigation.

### Duplicate-Based Analysis

The method assumes that repeated exact text is strong evidence of
automation. Sophisticated bots using varied wording would not
necessarily be detected by this approach.

------------------------------------------------------------------------

# 💡 Key Findings

The notebook produces several important observations:

### 1. Extremely High Repetition

**99.70%** of the 100,000 comments are classified as likely
bot-generated under the project's repetition-based heuristic.

### 2. Strong Structural Patterns

Repeated comments occur in unusually large and regular clusters,
including repetition counts around **997 and 1,994**.

### 3. Different Engagement Profiles

Likely genuine comments have much greater engagement variability and
include extremely highly liked comments, while likely bot comments show
a much tighter engagement range.

### 4. Sentiment Differences

Both groups contain substantial neutral sentiment, but the likely
genuine comments show a broader sentiment range.

### 5. Bot Comments Show Additional Behavioral Signals

The notebook identifies repeated viewpoints, multilingual repeated
comments, synchronized publication times, and repeated engagement values
as additional indicators of coordinated or automated behavior.

### 6. Potential Impact on Public-Sentiment Analysis

If the majority of comments are automated or duplicated, calculating
public sentiment directly from the complete dataset could produce a
misleading picture of genuine audience opinion.

This is arguably the central analytical lesson of the project:

> **Sentiment analysis should consider the authenticity and independence
> of the underlying observations before interpreting aggregate public
> opinion.**

------------------------------------------------------------------------

## 🔮 Possible Future Improvements

The project could be extended in several directions:

-   Build a supervised bot-detection classifier.
-   Incorporate author-level behavioral features.
-   Use publication-time patterns as explicit features.
-   Analyze comment similarity using TF-IDF or embeddings rather than
    exact matching alone.
-   Apply clustering to discover coordinated comment groups.
-   Use transformer-based sentiment models such as BERT-family
    architectures.
-   Perform multilingual sentiment analysis.
-   Detect sarcasm and irony.
-   Analyze temporal bursts of commenting activity.
-   Investigate network relationships between authors and comments.
-   Compare sentiment before and after filtering suspected automated
    activity.
-   Introduce human annotation to validate the bot-detection heuristic.
-   Quantify the difference between sentiment calculated on all comments
    and sentiment calculated only on likely genuine comments.

------------------------------------------------------------------------

## 🧭 Project Workflow

``` text
                ┌───────────────────────┐
                │   YouTube Comments     │
                │      100,000 rows      │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │   Data Quality Checks  │
                │ Shape / Nulls / Stats │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ Repetition Analysis   │
                │ Duplicate Text Counts │
                └───────────┬───────────┘
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
       ┌──────────────────┐   ┌──────────────────┐
       │ Likely Genuine   │   │   Likely Bots    │
       │   298 comments   │   │ 99,702 comments  │
       └────────┬─────────┘   └────────┬─────────┘
                │                      │
                └──────────┬───────────┘
                           ▼
                ┌───────────────────────┐
                │ Engagement Analysis   │
                │ Likes / Replies / Time │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ Text Preprocessing    │
                │ Tokenize / Stopwords  │
                │ Lemmatize             │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ NLTK VADER Sentiment  │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ Comparative Analysis  │
                │ Sentiment + Behavior  │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ Insights & Conclusion │
                └───────────────────────┘
```

------------------------------------------------------------------------

## 👤 Author

**Subhadeep Dey**

A data-science and NLP project exploring how automated content can
affect the interpretation of online public discourse.

------------------------------------------------------------------------

## ⭐ Conclusion

This project demonstrates why **data quality and source authenticity are
fundamental to NLP-based sentiment analysis**.

The analysis does not simply ask *"What sentiment do these 100,000
comments express?"* It first asks a more fundamental question:

> **"How many of these comments should be treated as independent
> expressions of user opinion?"**

By combining repetition analysis, engagement patterns, publication
behavior, and sentiment analysis, the project provides an exploratory
framework for investigating potential automated activity in online
political discourse.
