
# Text Similarity Recommendation System
This is a repository for Item RecSys models in Python. You can get the similar Items based on text similarity as follows.

- [Data Description](#data-description)
- [Process](#process)
- [Installation](#installation)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
  * [Example notebooks](#example-notebooks)
    + [Data Description](#data-description)
    + [Parameter Description](#parameter-description)
    + [Pipeline](#pipeline)
  * [Command Prompt](#command-prompt)
    + [Precautions <br>](#precautions--br-)
      - [1. yaml file](#1-yaml-file)
      - [2. json file](#2-json-file)
    + [Execute the file](#execute-the-file)
      - [To predict with newly-trained model](#to-predict-with-newly-trained-model)
      - [To predict with Pre-trained model](#to-predict-with-pre-trained-model)

---

# Data Description
#### Input
This model recommends items that are highly related to each item in `Items`, which means the source of the recommended items is also `Items`. If you add some text data related to the corresponding `Items` to `related_to_Items`(e.g., Items description, category, etc.), it helps to increase the model accuracy. 


```python
Items = [
          'Netflix movie',
          'Netflix party',
          'Netflix top',
          'Netflix ratings',
          'rotten tomatoes ratings',
          'IMDb Top 250 Movie ratings'
          ]
          
related_to_Items = [
          ["movie top", "Netflix"],
          ["party pricing", "Netflix"],
          ["top TV shows',","Netflix"],
          ["ratings"],
          ['tomatoes'],
          ['ratings']
          ]
```

#### Output

```markdown
Netflix movie
1: rotten tomatoes ratings
2: IMDb Top 250 Movie ratings
3: Netflix top

Netflix top
1: IMDb Top 250 Movie ratings
2: Netflix movie
3: Netflix ratings

IMDb Top 250 Movie ratings
1: Netflix ratings
2: Netflix top
3: Netflix movie
```

# Process

![video-TextSimila drawio](https://user-images.githubusercontent.com/48239962/171642236-46d3cbfc-92ca-47e5-9c14-93c9a9a86463.png)


**Tokenization**

extract nouns from each sentence

```{python}
# Example
['Netflix movie', 'Netflix party']
```
```
[['Netflix', 'movie'], ['Netflix', 'party']]
```

**Embedding**

get embedding vector from each sentence

```{python}
# Example
[['Netflix', 'movie'], ['Netflix', 'party']]
```
```
[[0.94, 0.13], [0.94, 0.741]]
```

After training tokenization and embedding models, the models are saved automatically. You can either train models with your own corpus or use the pre-trained models.

**Calculate cosine similarity**

calculate the similarity between item embedding vectors using cosine similarity.

$$
emb_A : \text{embedding vector of item A}\\
emb_B : \text{embedding vector of item B}\\
cos(emb_A,emb_B) = \frac{emb_A\cdot emb_B}{
\|emb_A\| \|emb_B\|}
$$


# Installation

```
pip install TextSimila
```

# Prerequisites
python version should be greater than 3.7.x 

```
pip install -r requirements.txt
```

# Quick Start

## Example notebooks
Refer to [`sample_code.ipynb`](https://github.com/dooboolab/TextSimila/blob/main/example/sample_code.ipynb) if you want to run code in a jupyter environment



### Parameter Description
The tables below describe the parameters of the class `text_sim_reco`

```
class text_sim_reco(
            Items,
            related_to_Items: list =  None,
            saved: Boolean = False,
            lang = Literal["en","ko"],
            reco_Item_number: int = 3,
            ratio: float = 0.3,

            # tokenize
            pretrain_tok: Boolean = False,
            stopwords: list = None,
            extranouns: list = None,
            verbose: Boolean = False,
            min_noun_frequency: int = 1,
            max_noun_frequency: int = 80,
            max_frequency_for_char: int = 20,
            min_noun_score: float = 0.1,
            extract_compound: Boolean = False,
            model_name_tok: str = None,
            
            # embedding
            pretrain_emb: Boolean = False,
            vector_size: int = 15,
            window: int = 3,
            min_count: int = 1,
            workers: int = 4,
            sg: Literal[1, 0] = 1,
            model_name_emb: str = None)
```



| Parameters                                                   | Attributes |
| ------------------------------------------------------------ | :--------- |
| **Items** : List[str] (required) |     A list of text data to recommend     |
  **related to Items** : List[List] (optional) |       A list of text data related to `Items` that helps to recommend  |
  **saved**: Boolean, default = False (optional) |    Whether to save the model       |
| **lang**: Literal["en","ko"], default = "en" |The configure model language<br />- 'ko': Your Items are in Koran <br />- 'en': Your Items are in English|
  **reco_Item_number** : int, default = 3 |The number of recommendations for each Item|
  **ratio**: float, default = 0.2 |    The minimum percentage that determines whether to create a corpus         


<br />
  
| Parameters for tokenization with Korean custom dataset                         | Attributes |
| ------------------------------------------------------------ | :--------- |
| **pretrain_tok**: Boolean, default = False  |      Whether to use Pre-trained model     |
  **min_noun_score** = float, default = 0.1   | The minimum noun score. It decides whether to combine single nouns and compounds |
  **min_noun_frequency** : int, default = 1   | The minimum frequency of words that occur in a corpus. It decides whether to be a noun while training(noun extracting) |
  **extract_compound** = boolean, default = False   |  Whether to extract compounds components <br />'compounds components': Information on single nouns that make up compound nouns
  **verbose**: boolean, default = False  | Whether to print out the current vectorizing |
  **stopwords** : List, default = None   | (Post-preprocessing option) A List of high-frequency of words to be filtered out   |
  **extranouns**: List, default = None   | (Post-preprocessing option) A List of nouns to be added  |
  **max_noun_frequency**: int, default = 80   | (Post-preprocessing option) The maximum frequency of words that occur in a corpus. It decides whether to be a noun after training |
  **max_frequency_for_char**: int, default = 20  | (Post-preprocessing option) `max_noun_frequency` option for words with length one  |
  **model_name_tok**: str = None   |      Pre-trained model name  |


<br />

| Parameters for embedding                                     | Attributes |
| ------------------------------------------------------------ | :--------- |
| **pretrain_emb**: Boolean, default = False |      Whether to use Pre-trained model     |
  **vector_size** : int, default = 15 |      Dimensionality of the word vectors     |
  **window**: int, default = 3 |     The maximum distance between the current and predicted word within a sentence     |
  **min_count**: int, default = 3 |      The model ignores all words with total frequency lower than this     |
  **workers**: int, default = 3 |      The number of worker threads to train     |
  **sg**: Literal[1, 0], default = 1 |     Training algorithm: skip-gram if sg=1, otherwise CBOW     |
  **model_name_emb**: str, default = None |      Pre-trained model name  |


---


## Command Prompt
By running `exe.py`, you can perform all the processes in `sample_code.ipynb` at once. Note that it **saves** the model and the predictions in the following format at every run

```
# Top3_prediction.json
{
  "Item_1": [
    "recommendation_1",
    "recommendation_2",
    "recommendation_3"
  ],

  ...

  "Item_10": [
    "recommendation_1",
    "recommendation_2",
    "recommendation_3"
  ]
}
```


### Precautions <br>
**Make sure that the following two files exist in the two folders below before executing `exe.py`**

1. yaml file in `config` folder
2. json file in `data` folder

#### 1. yaml file
If you want to adjust the hyperparameters, modify existing `model.yaml`. 

You can also create your own yaml file, but you must follow the existing `model.yaml` form and save it in `config` folder.


#### 2. json file
If you want to use your custom data, you must process and save it according to the format below. 

```
[
  {
      "Items": "Item_1",
      "related_to_Items": ["related_Items", "Item_1_discription"]
  },
  
  ...

  {
      "Items": "Item_10",
      "related_to_Items": ["Item_10_channel"]
  }

]
```


### Execute the file

#### To predict with newly-trained model

```
$ python exe.py [yaml_name] [file_name] --saved [saved]
```

#### To predict with Pre-trained model
※ If you want to use English custom dataset
```
$ python exe.py [yaml_name] [file_name] --pretrain_tok [pretrain_tok] --pretrain_emb [pretrain_emb]
```

To make it simpler, 

```
$ python exe.py [yaml_name] [file_name] -tok [pretrain_tok] -emb [pretrain_emb]
```

For example, 

#### Train ver.
```
# If you want to train the model without saving
$ python exe.py model.yaml sample_eng

# If you want to train the model and then save them
$ python exe.py model.yaml sample_eng --saved True
```

#### Pre-trained ver.
```
# If you want to use Pre-trained model for tokenization and embedding
$ python exe.py model.yaml sample_eng -tok True -emb True
```
