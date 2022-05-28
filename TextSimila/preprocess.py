import re
import os

# Get the number of the most recently saved file and the number of the newly saved file 

def getFileName(
        file_name: str, 
        extension: str,
        path: str):
    '''    
    - Inputs:
        file_name: file name
        extension: file extension
        path: path to browse the file
    
    - Outputs:
        recent_num: number of the most recently saved file name
        next_num: number of the newly saved file 

        Ex.) file_name: 'photo', extension: 'png', path: 'D:/'
        if there's no file that matches the condition in path
        recent_num: ''
        next_num: ''

        if the most recently saved file in path is 'photo.png'
        recent_num: ''
        next_num: '_(2)'

        if the most recently saved file in path is 'photo (2).png'
        recent_num: '_(2)'
        next_num: '_(3)'    
    '''
    
    file_list = list(filter(lambda x: file_name in x and os.path.splitext(x)[1] == '.'+extension, os.listdir(path)))
    if len(file_list)==0: return file_name+'.'+extension
    else:
        recent_file = max(map(lambda x: (os.path.getctime(os.path.join(path, x)), x), file_list))[1]
        recent_n = recent_file.split('(')[1].split(')')[0] if "(" in recent_file else ''
    if recent_n == '': recent_n = 1
    next_num = int(recent_n) + 1
    return file_name + f'_({next_num})' + '.' + extension

def preprocess(
        st: str,
        lang = 'ko'):
    '''
    - Inputs:
        st: text to be preprocesseed
        lang: language

    - Outputs:
        preprocessed text
    '''

    # Remove an web address
    st = ' '.join([x for x in st.split(' ') if not x.startswith('https://')])

    # Remove an Email
    st = re.sub('[a-zA-Z0-9_-]+@[a-z]+.[a-z]+[.][a-z]+',' ',st)

    # Remove a phone number
    st = re.sub('[\d]+-[\d]+-[\d]+',' ',st)

    # Extract only English, Korean, and numbers
    if lang == 'en':
        st = [re.sub('[^A-Za-z0-9]', '',x) for x in st.split()]
    if lang == 'ko':
        st = [re.sub('[^0-9가-힣]', '',x) for x in st.split()]
    
    st = ' '.join(st)

    # Remove an empty space
    st = st.replace('\r'," ")
    st = st.replace('\n'," ")
    st = st.replace('\t'," ")
    
    return st

########################################################################################################################################

def makeCorpus(
        data,
        ratio = 0.3,
        lang = 'ko'):
    '''
    - Inputs:
        data: data to make a corpus
        ratio: minimum percentage that determines whether to create a corpus
        lang: language
    
    - Outputs:
        Corpus
    '''

    corpus = []

    for st in data:
        pr = preprocess(st=st,lang=lang)
        if len(pr.split(' '))/len(st.split(' ')) > ratio: 
            corpus.append(pr)
        else:
            corpus.append('')

    return corpus

########################################################################################################################################

def maketrainCorpus(
        data,
        ratio: float = 0.3,
        lang = 'ko'):
    '''
    - Inputs:
        data: data to make a corpus
        ratio: minimum percentage that determines whether to create a corpus
        lang: language
    
    - Outputs:
        Corpus
    '''

    corpus = []

    for Items in data:
        corpus += makeCorpus(Items,ratio,lang)

    return corpus







