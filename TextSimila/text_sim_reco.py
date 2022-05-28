from xmlrpc.client import Boolean
from TextSimila import preprocess
import numpy as np
try:
    from typing import Literal # python >= 3.8
except:
    from typing_extensions import Literal 

from soynlp.tokenizer import NounLMatchTokenizer
from soynlp.noun import LRNounExtractor_v2
from gensim.models import FastText
import nltk
import os
import pickle

class Tokenize():

    def __init__(
            self,
            pretrain_tok: Boolean = False,
            lang: Literal['en','ko'] = 'eg',
            stopwords: list = None,
            extranouns: list = None,
            verbose: Boolean = False,
            min_noun_frequency: int = 1,
            max_noun_frequency: int = 80,
            max_frequency_for_char: int = 20,
            min_noun_score: float = 0.1,
            extract_compound: Boolean = False,
            model_name_tok: str = None,
            saved: Boolean = False 
            ):


        self.lang = lang
        self.pretrain_tok = pretrain_tok
        self.stopwords = stopwords
        self.extranouns = extranouns
        self.verbose = verbose
        self.min_noun_frequency = min_noun_frequency
        self.max_noun_frequency = max_noun_frequency
        self.max_frequency_for_char = max_frequency_for_char
        self.min_noun_score = min_noun_score
        self.extract_compound = extract_compound
        self.model_name_tok = model_name_tok
        self.saved = saved

        if lang not in ['en','ko']:
            raise AttributeError("language must be either 'ko' or 'en'")

        if lang == "en":
            if not stopwords or not extranouns:
                raise AttributeError("If you want to use English custom dataset, stopwords and extranouns must be None")

    
    def train(
            self,
            corpus: list
            ):
        
        self.model_path = './model'
        extension = 'pickle'
        

        # Korean custom dataset
        if self.lang=="ko":

            # Train new model
            if self.pretrain_tok==False:
                noun_extractor = LRNounExtractor_v2(
                        verbose=self.verbose,
                        extract_compound=self.extract_compound)

                nouns = noun_extractor.train_extract(
                        corpus,
                        min_noun_frequency=self.min_noun_frequency,
                        min_noun_score=self.min_noun_score)
                
                keys = list(nouns.keys())
                for key in keys:
                    if len(key) == 1 and nouns[key].frequency > self.max_frequency_for_char: nouns.pop(key)
                    elif nouns[key].frequency > self.max_noun_frequency: nouns.pop(key)

                if self.stopwords:
                    keys = list(nouns.keys())
                    for key in keys: 
                        if key in self.stopwords: nouns.pop(key)
                
                if self.extranouns:
                    for n in self.extranouns: 
                        nouns.update({n:1})

                self.nouns = nouns
                
                if self.saved==True:
                    if not os.path.isdir(self.model_path):
                        os.mkdir(self.model_path)
                    file_name = f'tokenized_nouns'
                    file_name = preprocess.getFileName(file_name, extension, self.model_path)

                    with open(os.path.join(self.model_path, file_name),'wb') as f:
                        pickle.dump(self.nouns, f)
                    print(f'The tokenized_nouns has been saved as "{file_name}" in "{self.model_path}"\n')
                else:
                    print(f'If you want to save these tokenized_nouns for pre-train, set saved option to be True\n')

            # Load pre-trained model
            elif self.pretrain_tok==True:
                if self.lang=="ko":
                    if not self.model_name_tok:
                        raise ValueError("Specify the `model_name_tok` to use in your yaml file")
                    else:
                        file_name = self.model_name_tok + '.' + extension
                        try:
                            with open(os.path.join(self.model_path, file_name), 'rb') as f:
                                print("Load pre-trained model for tokenization...\n")
                                self.nouns = pickle.load(f)
                        except:
                            raise FileNotFoundError(f"There's no Pre-trained model. Set `pretrain_tok` to be False")
        
        
        # English custom dataset
        elif self.lang=="en":
            nltk.data.path.append(self.model_path)
            nltk_data = [('tokenizers', 'punkt'), ('taggers', 'averaged_perceptron_tagger')]
            for folder, data in nltk_data:                
                if not folder in os.listdir(self.model_path):
                    # Train new model
                    if self.pretrain_tok==False:
                        nltk.download(data, quiet=True, download_dir=self.model_path)
                        print(f'NLTK data {data} has been saved in "{self.model_path}"')
                    # Load pre-trained model
                    elif self.pretrain_tok==True:
                        raise FileNotFoundError(f"There's no Pre-trained model. Set `pretrain_tok` to be False")
                else:
                    print(f'NLTK data {data} has been already saved in "{self.model_path}"')
                print()

        return self

    def token_by_sentence(
            self,
            sentence,
            lang):

        if lang == 'ko':
            tokenizer = NounLMatchTokenizer(self.nouns)
            token = tokenizer.tokenize(sentence)
                
        elif lang == 'en':
            words = nltk.tokenize.word_tokenize(sentence)
            token = [x[0] for x in nltk.tag.pos_tag(words) if x[1] == 'NN']
        
        return token

    def token_by_corpus(
            self,
            corpus: list,
            lang,
            ):

        token_list = [self.token_by_sentence(st,lang) for st in corpus]      
        return token_list

########################################################################################################################################       

class Text_sim_reco():

    def __init__(
            self,
            Items,
            related_to_Items: list =  None,
            saved: Boolean = False,
            lang: Literal["en","ko"] = "en",
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
            model_name_emb: str = None):
        
        self.Itemsnum = len(Items)
        self.lang = lang
        self.Items = Items
        self.related_to_Items = related_to_Items
        self.reco_Item_number = reco_Item_number
        self.ratio = ratio
        self.model_path = './model'
        self.saved = saved

        # token
        self.pretrain_tok = pretrain_tok
        self.stopwords = stopwords
        self.extranouns = extranouns
        self.verbose = verbose
        self.min_noun_frequency = min_noun_frequency
        self.max_noun_frequency = max_noun_frequency
        self.max_frequency_for_char = max_frequency_for_char
        self.min_noun_score = min_noun_score
        self.extract_compound = extract_compound
        self.model_name_tok = model_name_tok

        # embedding
        self.pretrain_emb = pretrain_emb
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.model_name_emb = model_name_emb


        self.params_token = {
                'pretrain_tok': self.pretrain_tok,
                'lang': self.lang,
                'stopwords':self.stopwords,
                'extranouns': self.extranouns,
                'verbose': self.verbose,
                'min_noun_frequency': self.min_noun_frequency,
                'max_noun_frequency': self.max_noun_frequency,
                'max_frequency_for_char': self.max_frequency_for_char,
                'min_noun_score': self.min_noun_score,
                'extract_compound': self.extract_compound,
                'model_name_tok': self.model_name_tok,
                'saved': self.saved}

        self.params_embedding = {
                'vector_size': self.vector_size,
                'window': self.window,
                'min_count': self.min_count,
                'workers': self.workers,
                'sg': self.sg}

        if len(self.related_to_Items) != self.Itemsnum:
            raise ValueError(f"'related_to_Items' and 'Itemsnum' have incompatible shapes.\n: {len(self.related_to_Items)} and {self.Itemsnum}")
        
        if sg not in [1, 0]:
            raise AttributeError("sg must be either 1(=skip-gram) or 0(=CBOW)")
        
        if lang not in ["en", "ko"]:
            raise AttributeError(f"It does not support the language {lang}")
        
        if lang == "en":
            if not stopwords or not extranouns:
                raise AttributeError("If you want to use Korean custom dataset, stopwords and extranouns must be None")

    
    def get_params_token(self):
        return self.params_token
    
    def get_params_embedding(self):
        return self.params_embedding

    def get_Items_from_index(self):
        return {i:Items for i,Items in enumerate(self.Items)}

    def get_Items_corpus(self):

        self.Items_corpus = preprocess.makeCorpus(data=self.Items,ratio=self.ratio,lang=self.lang)
        if self.related_to_Items:
            related_to_Items_corpus = preprocess.maketrainCorpus(data=self.related_to_Items,ratio=self.ratio,lang=self.lang)
            self.Items_corpus += related_to_Items_corpus
        
        return self

    def train(self):
        
        self.get_Items_corpus()
        
        embedding = Embedding(self)
        embedding.train()

        mean_vector = []
        for Items in self.Items:
            mean_vector.append(embedding.embedding(Items))
        mean_vector = np.array(mean_vector)

        cos_sim = np.dot(mean_vector,mean_vector.T)

        self.cos_sim = cos_sim

    def get_cos_sim_matrix(self):

        return self.cos_sim

    def predict(self):

        Items_info = self.get_Items_from_index()
        
        pred = []
        for i,score in enumerate(self.cos_sim):
            idx = sorted(range(len(score)), key=lambda k: score[k],reverse=True)[:self.reco_Item_number+1]
            if i in idx: idx.remove(i)
            else: idx = idx[:-1]
            pred.append([Items_info[i] for i in idx])

        return pred


########################################################################################################################################

class Embedding():

    def __init__(self, Text_sim_reco):

        self.Text_sim_reco = Text_sim_reco

    def train(self):
        self.model_path = './model'
        extension = 'pickle'

        # tokenize
        self.tokenize = Tokenize(**self.Text_sim_reco.params_token)
        self.tokenize.train(corpus=self.Text_sim_reco.Items_corpus)
        tokens = self.tokenize.token_by_corpus(corpus=self.Text_sim_reco.Items_corpus, lang=self.Text_sim_reco.lang)

        # embedding
        # Train new model
        if self.Text_sim_reco.pretrain_emb==False:
            self.embedding_model = FastText(tokens,**self.Text_sim_reco.params_embedding)
            
            # save the model
            if self.Text_sim_reco.saved==True:
                if not os.path.isdir(self.model_path):
                    os.mkdir(self.model_path)

                file_name = f'embedded_model'
                file_name = preprocess.getFileName(file_name, extension, self.model_path)
                
                with open(os.path.join(self.model_path, file_name),'wb') as f:
                    pickle.dump(self.embedding_model, f)
                print(f'The embedded_model has been saved as "{file_name}" in "{self.model_path}"\n')
            else:
                print(f'If you want to save these embedded model, set saved option to be True\n')

        # Load pre-trained model
        else:
            if not self.Text_sim_reco.model_name_emb:
                raise ValueError("Specify the `model_name_emb` to use in your yaml file")
            else:
                file_name = self.Text_sim_reco.model_name_emb + '.' + extension
                try:
                    with open(os.path.join(self.model_path, file_name), 'rb') as f:
                        print("Load pre-trained model for embedding...\n")
                        self.embedding_model = pickle.load(f)
                except:
                    raise FileNotFoundError(f"There's no Pre-trained model. Set `pretrain_emb` to be False")
                    

    def embedding(self, Items_name):

        mv = 0 ; cnt = 0
        tokens = self.tokenize.token_by_sentence(Items_name, self.Text_sim_reco.lang)
        if tokens == []:
            mv = np.random.random(self.Text_sim_reco.params_embedding['vector_size'])
        else:
            for token in tokens:
                cnt += 1
                mv += np.array(self.embedding_model.wv[token])
            mv /= cnt
        
        mv = mv/np.sqrt(sum(mv**2))

        return mv







