Prevendo a Utilidade de Comentários de Jogos em Português Brasileiro no Site Steam
================
Germano Antonio Zani Jorge

Instituto de Ciências Matemáticas e Computação, Universidade de São Paulo, Brasil.

e-mail: germano.jorge@alumni.usp.br
-------------
Citação:
Por favor, não se esqueça de citar o trabalho caso utilize-o:
-----------------
```
@article{jorge2023steambr,
  title={SteamBR: a dataset for game reviews and evaluation of a state-of-the-art method for helpfulness prediction},
  author={Jorge, Germano Antonio Zani and Pardo, Thiago Alexandre Salgueiro},
  journal={Anais},
  year={2023}
}
````
Nossos Objetivos:
---------

> Construir um modelo capaz de prever precisamente a utilidade de comentários de jogos.



> Investigar quais atributos tornam um comentário útil.

Ferramentas Utilizadas:
---------
- Python
- Pandas
- Sckit-learn
- Numpy
- Gensim
- Embeddings
- LDA (Latent Dirichlet Allocation)
- LIWC
- NLTK
- Matplotlib
- Seaborn

## Dados:
----------
Utilize os links para realizar o download do conjunto de dados e arquivos de suporte:
- **SteamBR corpus** [(https://drive.google.com/drive/folders/1TZynwUnYpKLOAGFmGnZkpuFtBpCgPKZx?usp=sharing)](https://drive.google.com/file/d/1aXolQJoZMC4Lt6LEyGofbvvGOxBR2Z4i/view?usp=drive_link)
- **reviews_filtradas** (https://drive.google.com/file/d/1tx_v1f3-SkfyC7tmVVW5wGBIAyPNNoDg/view?usp=drive_link)
- **meu_doc2vec** (https://drive.google.com/file/d/1bDS7Y3IX7irZP8Zz-7PHW1h2HQr6jeEV/view?usp=drive_link)
- **meu_lda** (https://drive.google.com/file/d/1Ox_qksC1pLmISYPYhCivG1ZvysmWyOPB/view?usp=drive_link)



Explicação dos arquivos:
------------------------

-   **steambrcorpus.zip:** Corpus com mais de 2 milhões de comentários em Português Brasileiro de jogos na Steam, extraídos de 10 mil jogos que tiveram seu nome e gênero anotados manualmente.


-   **reviews_filtradas.zip:** Contém mais de 230 mil comentários em Português Brasileiro retirados do site steam.com, após serem filtrados aqueles possuíam *3 votos ou mais*. Os comentários foram classificados e agrupados em 10 gêneros diferentes. Depois disso, foram dividos ao meio para que uma metade *(part_50)* fosse utilizada no treinamento dos vetores de documento (doc2vec) e a outra *(rest_part_50)* para o treino e teste do algoritmo. Dessa forma, neste arquivo .zip há um total de 10 diretórios cujos nomes se referem aos gêneros obtidos e que contêm 2 arquivos, *e.g.,* *Racing_json_part_50* e *Racing_json_rest_part_50*. Além disso, há também um diretório que contém as duas partes de todos os gêneros combinados.

-   **meu_doc2vec:** Modelo de vetores de documentos doc2vec (Le e Mikolov, 2014) com 1000 dimensões já treinado utilizando metade do conjunto de dados. Trata-se de uma representação das sentenças dos comentários através de vetores que permite com que sentenças com significados semelhantes possuam representações semelhantes. Para uma explicação mais detalhada, recomenda-se a leitura de Lau e Balwdwin (2016).

 -  **meu_lda:** Modelo de Latent Dirichlet Allocation (LDA) já treinado. Um modelo estatístico para a descoberta de tópicos abstratos. Blei DM, Ng AY, Jordan MI (2003)

 -  **LIWC2007_Portugues_win (1).dic:** Dicionário que contém palavras que revelam determinados sentimentos e opiniões (Balage Filho et al., 2013, Pennebaker et al., 2001)
 - **predUtil_h05_Racing.ipynb:** Script do código em formato de python notebook para a criação do algoritmo. Será utilizado como exemplo no tutorial.

Como reproduzir o resultado:
----------------------------
    OBS: Neste exemplo usaremos o Google Colab. Contudo, este procedimento pode ser reproduzido no Jupyter Notebooks com algumas alterações.
-   Baixe os dados pelos link do drive acima.
-   Mova os arquivos para seu drive em <https://drive.google.com>. Certifique-se de movê-los para um caminho que você possa lembrar posteriormente.
-   Inicie sua sessão em <https://colab.research.google.com>
-   No canto superior esquerdo, clique em "arquivo" -> "abrir notebook" e procure ou faça o upload do notebook *predUtil_h05_Racing.ipynb* contido neste repositório do github.
-   Procure no código por caminhos como *path = '/content/drive/MyDrive/Racing_json_part50.json'* e certifique-se de alterá-los para o local em que você depositou seus arquivos no drive anteriormente.
-  Execute o script

Tutorial de treinamento e avaliação do modelo:
------------------------------------------

Neste tutorial vamos investigar a utilidade de comentários no gênero Racing. Foram considerados apenas os comentários com número de votos maior ou igual a 3. O passo-a-passo a seguir é feito utilizando o script **predUtil_h05_Racing.ipynb**.


### 1. Carregar as Bibliotecas
``` python
from google.colab import drive
drive.mount('/content/drive')
!pip install unidecode
from unidecode import unidecode
import string
import liwc
import pandas as pd
import numpy as np
import glob
import re
from collections import Counter
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords
!pip install -U gensim
import gensim
from gensim.models import KeyedVectors
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
```
Clique em "aceitar" e coloque suas credencias para montar o seu drive.

### 2. Preparar o Doc2Vec

OBS: SE VOCÊ NÃO DESEJA TREINAR SEU PRÓPRIO DOC2VEC, PULE ESTA ETAPA.

``` r
#lê o arquivo de gêneros combinados
path = '/content/drive/MyDrive/Combined_json_part50.json'
df = pd.read_json(path)
```

``` r
#preprocessamento
from nltk.corpus import stopwords
def process_text(text):
  text = text.lower().replace('\n', ' ').replace('\r', '').strip()
  text = re.sub(' +',' ',text)
  text = re.sub(r'[^\w\s]','',text)
  text = re.sub('[0-9]+', '', text)

  stop_words = set(stopwords.words('portuguese'))
  word_tokens = word_tokenize(text)
  filtered_sentence = [w for w in word_tokens if not w in stop_words]




  text = ' '.join(filtered_sentence)
  return text
```

``` r
df = df['review'].dropna().apply(process_text) #tira os NaN e processa o texto
```

``` r
#define uma função para taggear os documentos para utiliza-los no doc2vec
def tagged_document(list_of_list_of_words):
  for i, list_of_words in enumerate(list_of_list_of_words):
    yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])
data_for_training = list(tagged_document(df))
```

``` r
model_dbow= gensim.models.doc2vec.Doc2Vec(vector_size=1000, min_count=2, epochs=30) #estabelece os parametros
model_dbow.build_vocab(data_for_training) #constroi o vocabulario para o modelo
model_dbow.train(data_for_training, total_examples=model_dbow.corpus_count, epochs=model_dbow.epochs) #treina o modelo
```

``` r
fname = '/content/drive/MyDrive/meu_doc2vec'
model_dbow.save(fname) #salva o modelo
````


### 3. Carregar o Doc2Vec pronto

``` r
fname = '/content/drive/MyDrive/meu_doc2vec'
model_dbow = Doc2Vec.load(fname)
````
### 4. Pré-processamento

``` r
#carrega o arquivo
path2 = '/content/drive/MyDrive/Tese/Racing_json_rest_part_50.json'
df2 = pd.read_json(path2)
```

Aqui vamos limpar o texto e normalizá-lo, colocando-o em minúsculo, excluindo números, pontuações e caracteres especiais.

``` r
#preprocessamento
def process_text(text):
  text = text.lower().replace('\n', ' ').replace('\r', '').strip()
  text = re.sub(' +',' ',text)
  text = re.sub(r'[^\w\s]','',text)
  text = re.sub('[0-9]+', '', text)

  stop_words = set(stopwords.words('portuguese'))
  word_tokens = word_tokenize(text)
  filtered_sentence = [w for w in word_tokens if not w in stop_words]
  
  text = ' '.join(filtered_sentence)
  return text
```

``` r
#transforma a coluna weighted vote score que era string em float para utilizar no algoritmo
df2 = df2.astype({'weighted_vote_score': float}, errors='raise')
```

Neste exemplo e no meu trabalho foram utilizados somente comentários com um score/threshold de utilidade maior do que 0.5

``` r
#cria uma coluna de utilidade com score maior do que 0.5
df2['Helpful'] = np.where(df2['weighted_vote_score'] > 0.5, True, False)

data = pd.DataFrame() #cria um dataframe


data['Text'] = df2['review'].dropna().apply(process_text) #tira os NaN e preprocessa o texto

data['Helpful'] = df2['Helpful']
```

``` r
stop_words = nltk.corpus.stopwords.words('portuguese')

def remove_stopwords(text,stop_words):
  
  # tudo para caixa baixa
  s = str(text).lower() 

  tokens = word_tokenize(s)

  # remove stopwords, dígitos, caracteres especiais e pontuações
  v = [word for word in tokens if not word in stop_words and word.isalnum() and not word.isdigit()]

  return v
  
df2 = data['Text']

textolimpo= df2.apply(lambda x:remove_stopwords(x, stop_words))
textolimpo

textolimpo.reset_index(drop=True, inplace=True)
```
o "textolimpo" remove as stopwords e tokeniza as sentenças.
Em seguida, vamos criar as colunas com os vetores de cada sentença

``` r
#cria uma lista contendo vetores inferidos a partir do doc2vec treinado
a = []
for i in textolimpo:
  b = model_dbow.infer_vector(i)
  a.append(b)
  
b = np.array(a) #transforma a lista num numpy array  
vetores = pd.DataFrame(b)

l = []
for i in range(1,1001):
  l.append(str('wv.'+ str(i)))
l

vetores.columns= l #transforma a lista de vetores em colunas no dataframe
```

Olhando o que fizemos até agora:

``` r
vetores
```

![vetores](https://github.com/germanojorge/PrevendoUtilidadeComentarios/blob/main/public/vetores.JPG)



``` r
dataframe = pd.DataFrame()
data.reset_index(drop=True, inplace=True) #tira o indice do dataframe criado anteriormente
resultfinal = pd.concat([data, vetores], axis=1, join='inner') #junta os dois dataframes
```

Depois de juntar os dois dataframes:
```
resultfinal
```
![dataframe](https://github.com/germanojorge/PrevendoUtilidadeComentarios/blob/main/public/resultfinal.JPG)


### 5. Preparar o LDA
OBS: CASO NÃO DESEJE PREPARAR SEU PRÓPRIO LDA, PULE ESTA ETAPA

``` r
#dicionariza
id2word = corpora.Dictionary(textolimpo)

corpus = []
for text in textolimpo:
    new = id2word.doc2bow(text)
    corpus.append(new)

#treina o LDA
from gensim.models.ldamulticore import LdaMulticore
lda_model = LdaMulticore(corpus=corpus,
                        id2word=id2word,
                        num_topics=30, 
                        random_state=100,
                        chunksize=100,
                        passes=10,
                        per_word_topics=True,
                        alpha = 0.9,
                        )
#salva o modelo
lda_model.save('/content/drive/MyDrive/meu_lda')                        
```                        
                    

### 6. Carregar o LDA
``` r
lda_model = LdaMulticore.load('/content/drive/MyDrive/meu_lda')

id2word = corpora.Dictionary(textolimpo)

corpus = []
for text in textolimpo:
    new = id2word.doc2bow(text)
    corpus.append(new)

#mostra os vetores pra cada review
train_vecs = []
for i in range(len(textolimpo)):
    top_topics = lda_model.get_document_topics(corpus[i], minimum_probability=0.0)
    topic_vec = [top_topics[i][1] for i in range(30)]
    train_vecs.append(topic_vec)
#distribuicao dos 30 topicos para cada review
``` 

``` r
l2 = []
for i in range(1,31):
  l2.append(str('topic.'+ str(i)))
l2



ldadf = pd.DataFrame(train_vecs)
ldadf.columns = l2
resultfinal = pd.concat([data, vetores, ldadf], axis=1, join='inner')
resultfinal
``` 







### 7. LIWC

``` r
!pip install liwc
parse, category_names = liwc.load_token_parser('/content/drive/MyDrive/LIWC2007_Portugues_win.dic')
listatexto = textolimpo.tolist()
textocounts = Counter(category for token in listatexto for category in parse(token))
listaliwc = []
for i in listatexto:
  a = Counter(category for token in i for category in parse(token))
  listaliwc.append(a)

dfliwc = pd.DataFrame(listaliwc).fillna(0)
resultfinal = pd.concat([data,vetores, ldadf, dfliwc], axis=1, join='inner')
resultfinal
``` 

### 8. Atributos de Metadados
Aqui estamos colocando colunas com o texto sujo apenas para referencia, e também renomeando algumas colunas.

``` r
path2 = '/content/drive/MyDrive/Tese/Racing_json_rest_part_50.json'
df2 = pd.read_json(path2)
df3 = df2.rename(columns={'review' : 'Text(dirty)'})
df4 = df3['Text(dirty)']
df4.reset_index(drop=True, inplace=True)
df2.rename(columns = {'voted_up':'Recommended'}, inplace = True)
df5 = df2['Recommended']
df5.reset_index(drop=True, inplace=True)
resultfinal = pd.concat([df5,df4,data, vetores, ldadf, dfliwc], axis=1, join='inner')
``` 
A ultima linha juntou todos os dataframes que criamos até agora: os vetores, o LDA e o LIWC.

- Em seguida, vamos preparar as varíaveis de metadados, criando uma coluna para cada

``` r
#trocando a variavel booleana por integral para depois colocar no algoritmo
resultfinal["Recommended"].replace({True: 1, False: 0}, inplace=True)
resultfinal["Helpful"].replace({True: 1, False: 0}, inplace=True)

#cria uma coluna para o numero de sentenças de cada review
resultfinal['n.sentences'] = resultfinal['Text(dirty)'].apply(sent_tokenize).tolist()
resultfinal['n.sentences'] = resultfinal['n.sentences'].apply(len)

#cria uma coluna com o numero total de palavras
resultfinal['n.words'] = [len(x.split()) for x in resultfinal['Text(dirty)'].tolist()]

#media do tamanho das sentenças
def avg_sentence_len(text):
  sentences = text.split(".") #split the text into a list of sentences.
  words = text.split(" ") #split the input text into a list of separate words
  if(sentences[len(sentences)-1]==""): #if the last value in sentences is an empty string
    average_sentence_length = len(words) / len(sentences)-1
  else:
    average_sentence_length = len(words) / len(sentences)
  return average_sentence_length #returning avg length of sentence

#aplica a uma coluna
resultfinal['avg.sentence.length'] = resultfinal['Text(dirty)'].apply(avg_sentence_len)

#numero de exclamações
def count_exclam(text):
  count = 0;  
  for i in text:
    if i in ('!'):  
        count = count + 1;  
          
  return count  

resultfinal['n.exclamation'] = resultfinal['Text(dirty)'].apply(count_exclam)


#numero de perguntas
def nquestion(text):
  a = len(re.findall(r'\?', text))
  return a

#aplicando
resultfinal['n.question'] = resultfinal['Text(dirty)'].apply(nquestion)

#proporcao de letras maiusculas
def capital_letters(text):
  try:
    a = sum(1 for c in text if c.isupper())/len(text)*100
  except ZeroDivisionError:
    a = 0
  return a

#aplicando
resultfinal['uppercase.ratio'] = resultfinal['Text(dirty)'].apply(capital_letters)
``` 





``` r
#procuramos os valores maximos para cada atributo e dividimos a coluna por eles para que os valores fiquem na mesma grandeza
max_value = resultfinal['n.sentences'].max()
resultfinal['n.sentences'] = resultfinal['n.sentences'].div(max_value)
max_value2 = resultfinal['n.words'].max()
resultfinal['n.words'] = resultfinal['n.words'].div(max_value2)
max_value3 = resultfinal['avg.sentence.length'].max()
resultfinal['avg.sentence.length'] = resultfinal['avg.sentence.length'].div(max_value3)
max_value4 = resultfinal['n.exclamation'].max()
resultfinal['n.exclamation']= resultfinal['n.exclamation'].div(max_value4)
max_value5 = resultfinal['n.question'].max()
resultfinal['n.question'] = resultfinal['n.question'].div(max_value5)
max_value6 = resultfinal['uppercase.ratio'].max()
resultfinal['uppercase.ratio'] = resultfinal['uppercase.ratio'].div(max_value6)
``` 

Este dataframe contém tudo o que fizemos até agora

![resultsuperficie](https://github.com/germanojorge/PrevendoUtilidadeComentarios/blob/main/public/resultsuperficie.JPG)




### 9. Quadro de atributos
Vamos olhar melhor todos os atributos que extraímos:

![features](https://github.com/germanojorge/PrevendoUtilidadeComentarios/blob/main/public/tabela_atributos.PNG)




### 10. Separando em features e target

``` r
features= pd.DataFrame(resultfinal.drop(columns=['Text(dirty)', 'Text', 'Helpful']))
target = pd.DataFrame(resultfinal['Helpful'])
``` 



### 11. Balanceamento
Faremos o baleanceamento para as classes (útil ou não-útil) utilizando a técnica STOMENN
``` r
#separa o treino e o teste
X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=0, test_size=0.2)

#realiza o under e oversampling para os dados ficarem balanceados
smote_enn = SMOTEENN(random_state=0)
X_train_res, y_train_res = smote_enn.fit_resample(X_train, y_train)


ax = y_train_res.value_counts().plot.pie(autopct='2%f')
_ = ax.set_title("Combined_sampling")
```

Gráfico com os dados balanceados:
![smote](https://github.com/germanojorge/PrevendoUtilidadeComentarios/blob/main/public/graficosmote.JPG)



### 12. Treinar o modelo de Classificação

```r
model_gbm = GradientBoostingClassifier(n_estimators=600,
                                       learning_rate=0.05,
                                       max_depth=3,
                                       subsample=0.5,
                                       validation_fraction=0.1,
                                       n_iter_no_change=20,
                                       verbose=1)
model_gbm.fit(X_train_res, y_train_res)
``` 


``` r
model_prediction = model_gbm.predict(X_test)
print('accuracy %s' % accuracy_score(model_prediction, y_test))
print(classification_report(y_test, model_prediction))
``` 
![classifc](https://github.com/germanojorge/PrevendoUtilidadeComentarios/blob/main/public/acc_class.JPG)

> Nosso modelo atingiu uma acurácia de 84%, e previu corretamente a classe 1 em 91%


### 13. Treinar o modelo de Regressão

``` r
target_reg = df2['weighted_vote_score']
#separa o treino e o teste
X_train, X_test, y_train, y_test = train_test_split(features, target_reg, random_state=0, test_size=0.2)

reg = GradientBoostingRegressor(n_estimators=600,
                                       learning_rate=0.05,
                                       max_depth=3,
                                       subsample=0.5,
                                       validation_fraction=0.1,
                                       n_iter_no_change=20,
                                       verbose=1)
reg.fit(X_train, y_train)


rmse = mean_squared_error(y_test, reg.predict(X_test), squared = False)
print("The Root mean squared error (RMSE) on test set: {:.4f}".format(rmse))
```

![reg](https://github.com/germanojorge/PrevendoUtilidadeComentarios/blob/main/public/rmse_reg.JPG)

> A raiz do erro quadrático médio atingiu valores de 0.09.

```r
y_pred = reg.predict(X_test)
df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
print(df_preds)
```
Se rodarmos um teste de predição, veremos que os valores são parecidos.

![reg2](https://github.com/germanojorge/PrevendoUtilidadeComentarios/blob/main/public/predict.JPG)

--------------------------------------

### 14. Verificar a importância dos atributos

``` r
feat_imp_class = pd.DataFrame(model_gbm.feature_importances_)
feat_imp_reg = pd.DataFrame(reg.feature_importances_)


feat_imp_class.nlargest(n=10, columns=[0])
feat_imp_reg.nlargest(n=10, columns=[0])
```

Faremos um dataframe para a importância das features de CLASSIFICAÇÃO
```r
transposto = features.transpose()
lista_index = [transposto.index]
index_df = pd.DataFrame(lista_index)
df_feat = index_df.transpose()
df_valor_feat_class = pd.concat([feat_imp_class, df_feat], axis=1)
df_valor_feat_class
``` 
``` r
df_valor_feat_class.columns=['valor', 'feature'] #renomeando as colunas
df_valor_feat_class
df_valor_feat_class.nlargest(n=10, columns=['valor'])
df_valor_feat_class.index.name= 'num_feat'
classfeat = df_valor_feat_class.nlargest(n=10, columns=['valor'])
classfeat
sns.barplot(data=classfeat, x='valor', y='feature',)
``` 
![reg](https://github.com/germanojorge/PrevendoUtilidadeComentarios/blob/main/public/grafico.JPG)

O gráfico nos indica que os atributos mais importantes para a predição de utilidade são
-   Atributos de metadados
-   Atributos de vetores de documento (Doc2Vec)
-   Atributos baseados em modelagem de tópico (LDA)
