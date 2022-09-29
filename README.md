Prevendo a Utilidade de Comentários de Jogos em Português Brasileiro no Site Steam
================
Germano Antonio Zani Jorge

Instituto de Ciências Matemáticas e Computação, Universidade de São Paulo, Brasil.

Nossos Objetivos:
---------

> Construir um modelo capaz de prever precisamente a utilidade de comentários de jogos.



> Investigar quais atributos tornam um comentário útil.


Explicação dos arquivos:
------------------------

-   **reviews_filtradas.zip:** Contém mais de 230 mil comentários em Português Brasileiro retirados do site steam.com, após serem filtrados aqueles possuíam *3 votos ou mais*. Os comentários foram classificados e agrupados em 10 gêneros diferentes. Depois disso, foram dividos ao meio para que uma metade *(part_50)* fosse utilizada no treinamento dos vetores de documento (doc2vec) e a outra *(rest_part_50)* para o treino e teste do algoritmo. Dessa forma, neste arquivo .zip há um total de 10 diretórios cujos nomes se referem aos gêneros obtidos e que contêm 2 arquivos, *e.g.,* *Action_json_part_50* e *Action_json_rest_part_50*. Além disso, há também um diretório que contém as duas partes de todos os gêneros combinados.

-   **meu_doc2vec:** Modelo de vetores de documentos doc2vec (Le e Mikolov, 2014) com 1000 dimensões já treinado utilizando metade do conjunto de dados. Trata-se de uma representação das sentenças dos comentários através de vetores que permite com que sentenças com significados semelhantes possuam representações semelhantes. Para uma explicação mais detalhada, recomenda-se a leitura de Lau e Balwdwin (2016).

 -  **meu_lda:** Modelo de Latent Dirichlet Allocation (LDA) já treinado. Um modelo estatístico para a descoberta de tópicos abstratos. Blei DM, Ng AY, Jordan MI (2003)

 -  **LIWC2007_Portugues_win (1).dic:** Dicionário que contém palavras que revelam determinados sentimentos e opiniões (Balage Filho et al., 2013, Pennebaker et al., 2001)

Como reproduzir o resultado:
----------------------------
obs: neste exemplo usaremos o Google Colab. Contudo, este procedimento pode ser reproduzido no Jupyter Notebooks com algumas alterações.
-   Baixe o repositório em <https://github.com/germanojorge/PrevendoUtilidadeComentarios>. Clique em "código" -> "download ZIP".
-   Mova os arquivos para seu drive em <https://drive.google.com>. Certifique-se de movê-los para um caminho que você possa lembrar posteriormente.
-   Inicie sua sessão em <https://colab.research.google.com>
-   Execute the script: gbm\_eval.R

Tutorial of model training and evaluation:
------------------------------------------

On the steam store there are many game genres e.g. action, racing, survival, rpg etc. We have analysed top 10 game genres that are discussed and explained in the paper. In this case, we are investigating the helpfulness of Racing game genre and showing the process to reproduce its result. We have considered those reviews which have at least 50 votes. The following step by step explanation are from the script **gbm\_eval.R.**

``` r
#Define some variables
genre <- "Racing"
vote_num <- 50
```

### 1. Load dataset

``` r
#load dataset 
dataFile <- paste0("Dataset/Reviews_", genre ,"_50.Rdata")
if(!file.exists(dataFile) || file.size(dataFile) == 0){
  stop("Data file doesn't exist")
}
load(dataFile)
original_dataset <- ds.reviews
```

### 2. Summarize dataset

Now it is time to take a look at the data

``` r
# dimensions of dataset
dim(original_dataset)
```

    ## [1] 2251 2205

You can see 2251 reviews and 2205 attributes in the original dataset of Racing genre.

### 3. Add target variable/column

Steam helpfulness (target variable) of a review is defined by the rating score (here, ws.score) of that review.
<!--- The rating score of a particular review is computed taking the lower bound of Wilson scoreconfidence interval for a Binomial parameter. In this case, we have used an R function "binom.confint" from the "binom" package.

``` r
##calculate Wilson score (rating)
##x represents the number of people who feel the review is helpful (Yes votes)
##n represents the total number of votes (i.e. sum of Yes and No votes)
original_dataset$ws.score <- binom.confint(x=original_dataset$vote.agree, n=original_dataset$vote.total, conf.level = 0.95, methods=c("wilson"))$lower
``` -->
Now depending on a score threshold (0.90) we defined our target variable helpful as follows:

``` r
##Add a helpful column
score_thrsld <- 0.90
original_dataset$helpful <- ifelse(original_dataset$ws.score > score_thrsld, "Yes", "No")
original_dataset$helpful <- as.factor(original_dataset$helpful)
dim.y <- "helpful"
```

### 4. Feature Selection

After doing the feature ranking (Rscript feature\_ranking.R), we have total **1789** features that as shown in the following table. Some insignificant features are removed during the ranking process.

<table style="width:100%;">
<colgroup>
<col width="13%" />
<col width="28%" />
<col width="57%" />
</colgroup>
<thead>
<tr class="header">
<th>Feature Type</th>
<th>Examples</th>
<th>Explanation</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Metadata (39)</td>
<td>e.g. recommend, days.interval, #words, #sentences</td>
<td>Text, Reviewer, Game metadata. Counting, Aggregating, Ratio variables</td>
</tr>
<tr class="even">
<td>LIWC (93)</td>
<td>liwc.* (e.g. liwc.Segment, liwc.WC, liwc.Analytic, liwc.Tone)</td>
<td>Linguistic Inquiry and Word Count (LIWC) reads a given text and counts the percentage of words that reflect different emotions, thinking styles, social concerns, and even parts of speech</td>
</tr>
<tr class="odd">
<td>TF-IDF (629)</td>
<td>tfidf.* (e.g. tfidf.die, tfidf.easy, tfidf.better, tfidf.bad)</td>
<td>Unigram term-weighting method used to extract features based on the word frequencies in the review and review corpus</td>
</tr>
<tr class="even">
<td>LDA (28)</td>
<td>topic.* (e.g. topic.1, topic.2, topic.3)</td>
<td>Topic based features induced from review corpus</td>
</tr>
<tr class="odd">
<td>Word2Vec (1000)</td>
<td>wv.* (e.g. wv.1, wv.2, wv.3, wv.4)</td>
<td>Word2vec, a group of related models that are used to produce word embeddings features from the text</td>
</tr>
</tbody>
</table>

``` r
#Load feature file
featureFile <- paste0("Features/FT_", genre, "_V", vote_num, "_R", score_thrsld,"_S25.Rdata")
if(!file.exists(featureFile)){
  stop("\nFeature File Not Exists!!")
}
load(featureFile)

#Total number of features
max.dims <- NROW(feature.rank)
print(max.dims)
```

    ## [1] 1789

``` r
#Total feature list
dim.x <- feature.rank[order(feature.rank$total)[1:max.dims],]$name

#Exclude tfidf features, we found there are not important at all
dim.x <- grep("^tfidf.*?", dim.x, value = TRUE,  invert = TRUE)
dataset <- original_dataset[, c(dim.x, dim.y), with=F]
#Peek at the Data   
head(dataset[,1:5], 5)
```

    ##    recommend   topic.17     wv.613     wv.367     wv.638
    ## 1:         0 0.04912281 -0.0196526 0.00585691 0.00564988
    ## 2:         0 0.04204204 -0.0181740 0.00550578 0.00636852
    ## 3:         0 0.04301075 -0.0237652 0.00789483 0.01222250
    ## 4:         1 0.02873563 -0.0156660 0.00388676 0.01468580
    ## 5:         1 0.02487562 -0.0175165 0.00381155 0.00345860

We did an investigation with a smaller sample size (50% of the dataset) taking the upper limit of feature size by 1000. The results are displayed in the following Figure.

<img src="figure/feature_score.png" title="F-score by number of features" alt="F-score by number of features" style="display: block; margin: auto;" /> 

From the above plot of F-score against number of feature, we can clearly see that the best F-scores of the dataset were found when the number of features were 840.

``` r
#Select number of features
gbm.dim <- 840
#Selec the best subset of the total features
gbm.dim.x <- dim.x[1:gbm.dim]
```

### 5. Balance the dataset

Let’s now take a look at the number of instances (rows) that belong to each class and balance the dataset.

``` r
#Total number of rows in the dataset
n.rows <- NROW(dataset)
#Class Distribution
print(table(dataset$helpful))
```

    ## 
    ##   No  Yes 
    ## 2187   64

Creates possibly balanced samples by random over-sampling minority examples, under-sampling majority examples or combination of over- and under-sampling.

``` r
#Balance data set with both over and under sampling
n.samplePercent <- 100
n.sampleSize <- ceiling(n.rows * n.samplePercent/100)
balanced_data <- ovun.sample(helpful ~ ., data = dataset, method="both", p=0.5, N=n.sampleSize)$data
print(table(balanced_data$helpful))
```

    ## 
    ##   No  Yes 
    ## 1105 1146

### 6. Partition into training and test dataset

Partition dataset into training (80%) and test (20%).

``` r
#Split dataset
split <- 0.80
trainIndex <- as.vector(createDataPartition(y=balanced_data$helpful, p=split, list=FALSE))

#Get train data
gbm.dataTrain <- balanced_data[trainIndex, c(gbm.dim.x,dim.y), ]
dim(gbm.dataTrain)
```

    ## [1] 1801  841

``` r
#Get test data
gbm.dataTest <- balanced_data[-trainIndex, c(gbm.dim.x,dim.y), ]
dim(gbm.dataTest)
```

    ## [1] 450 841

``` r
#Split train data
gbm.trainX <- gbm.dataTrain[, gbm.dim.x, ]
gbm.trainY <- as.factor(gbm.dataTrain[, dim.y, ])
        
#Split test data 
gbm.testX <- gbm.dataTest[, gbm.dim.x, ]
gbm.testY <- as.factor(gbm.dataTest[, dim.y, ])
```

### 7. Preprocess data

``` r
#Remove columns with near zero variance
nearZV <- nearZeroVar(gbm.trainX)
if(length(nearZV) > 0){
    gbm.trainX <- gbm.trainX[, -nearZV]
    gbm.testX <- gbm.testX[, -nearZV]
}

#Preprocess training Data
preObj <- preProcess(gbm.trainX, method = c("center", "scale"))
gbm.trainX <- predict(preObj, gbm.trainX)
        
#Preprocess test Data
preObj <- preProcess(gbm.testX, method = c("center", "scale"))
gbm.testX <- predict(preObj, gbm.testX)
```

### 8. Train and build the model

We will apply 10-fold crossvalidation to build the model. In our experiment, we have used more reliable evaluation metrics: F-score and AUC. We have employed a GBM (Gradient Boosting Machine) classifier.

``` r
#Control parameters 
gbm.fitControl = trainControl(method="repeatedcv", #small size -> repeatedcv
                              number=10, #10-fold cv
                              repeats=3,
                              returnResamp = "final",
                              selectionFunction = "best",
                              classProbs=TRUE, 
                              summaryFunction=twoClassSummary,
                              allowParallel = TRUE)
#Train the model
gbmFit <- train(gbm.trainX, gbm.trainY, method="gbm", metric="ROC", trControl=gbm.fitControl, verbose=F)
```

### 9. Evaluate the model

``` r
#Get evaluation score
eval_score <- get_eval_score(gbmFit, gbm.trainX, gbm.trainY, gbm.testX, gbm.testY, score_thrsld, "GBM")
#print(eval_score)
```

We conducted a series of executions (10 times) from the steps 5-8 in model training and the testing process to make the results more accurate. The evaluation results are saved in a file in the Evaluation directory. The mean evaluation metrics are shown in the following table.

``` r
scoreFile <- paste0("Evaluation/SCORE_", genre,"_V50_R0.9.Rdata")
load(scoreFile)
final_eval_score <- sapply(Filter(is.numeric, comb.score), mean)
```

| Evaluation Metric | Training score | Test score |
|-------------------|----------------|------------|
| Accuracy          | 1              | 0.989      |
| Precision         | 1              | 0.988      |
| Recall            | 1              | 0.991      |
| F1-score          | 1              | 0.989      |
<!---
``` r
#Plot AUC curve
test.pred <- predict(gbmFit, gbm.testX, type="prob")
test.ref <- gbm.testY
predob <- prediction(test.pred$Yes, test.ref)
perf <- performance(predob, "tpr", "fpr")
plot(perf)
``
![](README_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-17-1.png)
-->

From the above evaluation result, we found our model performed very well to classify whether the reviews were helpful or not. All evaluation metrics are about 99% for test dataset.

<!---
### 10. Model Validation: Compare with the null model

One way of validating our model is to compare the model with the null model. We have a seperate script **gbm\_nullModel.R**. By executing this file we have the following resrults. Remember that in null model, all observations are predicted as the major class. In our case it is No.

``` r
#Confusion matrix of the null model from training dataset
null.cmTrain <- as.table(array(c(1750, 0, 52, 0),dim=c(2,2), dimnames=list(c("No","Yes"),c("No","Yes"))))
names(attributes(null.cmTrain)$dimnames) <- c("Reference","Prediction")
print(null.cmTrain)
```

    ##          Prediction
    ## Reference   No  Yes
    ##       No  1750   52
    ##       Yes    0    0

``` r
#Confusion matrix of the null model from test dataset
null.cmTest <- as.table(array(c(437, 0, 12, 0),dim=c(2,2), dimnames=list(c("No","Yes"),c("No","Yes"))))
names(attributes(null.cmTest)$dimnames) <- c("Reference","Prediction")
print(null.cmTest)
```

    ##          Prediction
    ## Reference  No Yes
    ##       No  437  12
    ##       Yes   0   0

| Evaluation Metric | Our model | Null model |
|-------------------|-----------|------------|
| Training Accuracy | 1         | 0.971      |
| Test Accuracy     | 0.989     | 0.973      |

From the above table, we can see that our model performed better than the null model.
-->

### 10. What makes review helpful

``` r
#Important features
gbmImp <- varImp(gbmFit, scale = TRUE)
impFeatures <- gbmImp$importance

#Sort by overall weight
impFeatures <- impFeatures[order(-impFeatures$Overall), , drop = FALSE]
head(impFeatures, 10)
```

    ##                    Overall
    ## wv.698           100.00000
    ## recommend         79.68782
    ## topic.17          68.56388
    ## wv.903            47.82878
    ## wv.819            41.77297
    ## wv.226            37.71184
    ## wv.701            36.39785
    ## wv.330            35.35896
    ## wv.512            32.78374
    ## n.content.remove  32.10252

Similar to the evaluation metrics, feature importance is also computed by averaging the weights of 10 executions and saved it in a file in the ImpFeatures directory. The plotting of feature importace is shown as follows.

``` r
#Load feature importace file
load(paste0("ImpFeatures/Imp_FT_", genre,"_V50_R0.9.Rdata"))

#Select top 20
top_features <- feature.imp[1:20, ]
top_features$feature <- factor(top_features$feature, levels=unique(top_features$feature))
featurePlot <- ggplot(data=top_features, aes(x=feature, y=mean )) +
  geom_bar(stat="identity", position=position_dodge(0.6), fill="steelblue", width=0.6) +
  scale_x_discrete(limits = rev(levels(top_features$feature))) +
  scale_y_continuous(breaks=c(seq(0,100,10))) +
  xlab("Features") + ylab("Feature importance (Relative weight)") + # Set axis labels
  theme_bw()+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())

#Horizontal bar plot
featurePlot <- featurePlot + coord_flip()
print(featurePlot)
```

![](README_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-21-1.png)

We found that review helpfulness mostly depends on:

-   Metadata features
-   The hidden word embedding features (Word2Vec) and
-   LDA topic based features.
