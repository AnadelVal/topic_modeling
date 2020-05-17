library(dplyr)
library(tidyverse)
library(tm)
library(stringr)
library(topicmodels)
library(wordcloud)
library(rmarkdown)
library(dendextend)
library(slam)
library(factoextra)
library(NbClust)
library(stringi)
library(cld2)
library(textmineR)
library(reshape)
library(data.table)

set.seed(1234) 

c <- read.csv("sample.csv",sep=",", header=TRUE) #download report from SNow and place it in the same folder as setwd() above

df <- data.frame(c) %>% select(1,5) 
df$text <- as.character(df$text)

# =============== FILTER BY ENGLISH ===========================
df$language <- detect_language(df$text)
df <- filter(df, language=="en")
df <- df %>% select(1,2) 

# ============== REMOVE CARRIAGE RETURNS, E-MAILS, PUNTUACTION SIGNS, PHONE NUMBERS, ATTACHMENTS, SERVERS ============================
#df['text'][1:5,]
df$text <- tolower(df$text)
df$text <- str_replace_all(df$text, "@" , " ") 
df$text <- str_replace_all(df$text, "[\n]" , " ")   # to remove carriage returns
df$text <- str_replace_all(df$text, "[\t]" , " ")   # to remove tabulation
df$text <- str_replace_all(df$text, ";" , " ")      # to remove ";"
df$text <- str_replace_all(df$text, "€" , " ")      # to remove ";"
df$text <- gsub("(http)\\S+"," ", df$text)  # to remove web pages
df$text <- gsub('\\S+\'\\S+', '', df$text) #to remove I've ...
df$text <- gsub("[-|_|+|=|/|~|^|ðÿ™|ð|ÿ|™|˜|š|â]"," ", df$text)  # to remove "-" or "_" or "+" or "=" pr "/" or "~" signs 
df$text <- str_replace_all(df$text, pattern = "[[:punct:]]", " ") #to remove puntuation
df$text<- gsub('\\b[[:alnum:]]{20,}\\b'," ", df$text) #to remove alphanumeric with length greater than 20 characters.
df$text<- gsub('\\S+[[:digit:]]\\S+'," ", df$text) #to remove numbers
df$text<- gsub('[[:digit:]]'," ", df$text) #to remove numbers

# ============= STOPWORDS ======================= to remove stopwords defined by default in library "SMART" and you can add personalized stopwords to be removed
df$text <- tm::removeWords(df$text, c(stopwords(kind = "SMART"),"mon","tue","wed","thurs","fri","sat","sun","jan","feb","mar","apr","may","jun","jul","aug","sept","oct","nov","dec",
                                                                    "monday","tuesday","wednesday","thursday","friday","saturday","sunday","january","february","march","april","june","july","august","september",
                                                                    "october","november","december","rr","mm","ay"))

# ============= REMOVE MULTIPLE SPACES =======================
df$text <- gsub("(?<=[\\s])\\s*|^\\s+|\\s+$", "", df$text, perl=TRUE)  # merge multiple spaces to single space

# ============= STEMMING ======================= to remove suffixes and keep only root
#Added plural to stopwords in order to avoid that stemming adds new words such as "user" or "mail"
df$text <- stemDocument(df$text, language = "english") # word stemming

# ========== TOPIC MODELING: LDA + hclust ==========

dtm <- CreateDtm(df$text, df$tweet_id) #Create DocumentTermMatrix, which is a matrix that lists all occurrences of words in the corpus. 
inc_ids <- rownames(dtm)

tf <- TermDocFreq(dtm = dtm)
original_tf <- tf %>% select(term, term_freq,doc_freq)
rownames(original_tf) <- 1:nrow(original_tf)

vocabulary <- tf$term[ tf$term_freq > 3 & tf$doc_freq < nrow(dtm) / 2 ] #keep only those words appearing more than 3 times and less than a half.


##### 1) EXPLORE THE DATA TO DECIDE THE OPTIMAL NUMBER OF CATEGORIES

# 20 models creation. A folder called "modelsxxxxx" will be created only the first time (for Windows machines). Be careful, if you change something relevant in the preprocessing, you should remove this folder and genreate the 20 models again.
k_list <- seq(1, 20, by = 1)
model_dir <- paste0("models_", digest::digest(vocabulary, algo = "sha1"))
if (!dir.exists(model_dir)) dir.create(model_dir)
model_list <- TmParallelApply(X = k_list, FUN = function(k){
  filename = file.path(model_dir, paste0(k, "_topics.rda"))
  
  if (!file.exists(filename)) {
    m <- FitLdaModel(dtm = dtm, k = k, iterations = 500)
    m$k <- k
    m$coherence <- CalcProbCoherence(phi = m$phi, dtm = dtm, M = 5)
    save(m, file = filename)
  } else {
    load(filename)
  }
  
  m
}, export=c("dtm", "model_dir")) # export only needed for Windows machines


# explore coherence matrix to decide the optimal number of categories. The peak shows the maximum of coherence of putting the top words together.
coherence_mat <- data.frame(k = sapply(model_list, function(x) nrow(x$phi)), 
                            coherence = sapply(model_list, function(x) mean(x$coherence)), 
                            stringsAsFactors = FALSE)
ggplot(coherence_mat, aes(x = k, y = coherence)) +
  geom_point() +
  geom_line(group = 1)+
  ggtitle("Best Topic by Coherence Score") + theme_minimal() +
  scale_x_continuous(breaks = seq(1,20,1)) + ylab("Coherence")

cat("Maximum coherence is reached with ", which.max(coherence_mat$coherence), " clusters")

# take the model fitted with the 20 categories
model_20topics <- model_list[20][[ 1 ]]
model_20topics$top_terms <- GetTopTerms(phi = model_20topics$phi, M = 20) #top 20 values per word in each topic
top20_wide <- as.data.frame(model_20topics$top_terms)
top20_wide


# hierarchical clustering: distances generated by LDA algorithm are the input of the clustering.
model_20topics$topic_linguistic_dist <- CalcHellingerDist(model_20topics$phi)
model_20topics$hclust <- hclust(as.dist(model_20topics$topic_linguistic_dist), "ward.D")
model_20topics$hclust$labels <- paste(model_20topics$hclust$labels, model_20topics$labels[ , 1])
plot(model_20topics$hclust) #plot tree diagram to see distribution of 20 topics.
# consumer: specify where to cut the tree though the parameter k (k = consumer_num_cluster). 
# This value can be infered from the tree diagram, the coherence plot or based on the consumer expertise. Default: 12. 
consumer_num_clusters = 11
rect.hclust(model_20topics$hclust, k = consumer_num_clusters-1, border = 2:10)
pdf("dendogram.pdf")
rect.hclust(model_20topics$hclust, k = consumer_num_clusters-1, border = 2:10)
dev.off()

##### 2) GENERATE THE FINAL MODEL BASED ON consumer PARAMETRIZATION & THE WORDCLOUD PER CLUSETR

model_final <- model_list[consumer_num_clusters][[ 1 ]]
model_final$top_terms <- GetTopTerms(phi = model_final$phi, M = 20) #top 20 values per word in each topic
top20_wide <- as.data.frame(model_final$top_terms)

#word, topic relationship ---------------------------------------------
#looking at the terms allocated to the topic and their pr(word|topic)
allterms <-data.frame(t(model_final$phi))
allterms$word <- rownames(allterms)
rownames(allterms) <- 1:nrow(allterms)
allterms <- melt(allterms,idvars = "word") 
allterms <- allterms %>% rename(c("variable" = "topic"))  
FINAL_allterms <- allterms %>% group_by(topic) %>% arrange(desc(value))

#Topic,word,freq ------------------------------------------------------
final_summary_words <- data.frame(top_terms = t(model_final$top_terms))
final_summary_words$topic <- rownames(final_summary_words)
rownames(final_summary_words) <- 1:nrow(final_summary_words)
final_summary_words <- final_summary_words %>% melt(id.vars = c("topic"))
final_summary_words <- final_summary_words %>% rename(c("value" = "word")) %>% select(-variable)
final_summary_words <- left_join(final_summary_words,allterms)
final_summary_words <- final_summary_words %>% group_by(topic,word) %>%
  arrange(desc(value))
final_summary_words <- final_summary_words %>% group_by(topic, word) %>% filter(row_number() == 1) %>% 
  ungroup() %>% tidyr::separate(topic, into =c("t","topic")) %>% select(-t)
word_topic_freq <- left_join(final_summary_words, original_tf, by = c("word" = "term"))

#wordcloud
pdf("wordcloud_per_cluster_definitive.pdf")
for(i in 1:length(unique(final_summary_words$topic)))
{  wordcloud(words = subset(final_summary_words ,topic == i)$word, freq = subset(final_summary_words ,topic == i)$value, min.freq = 1,
             max.words=200, random.order=FALSE, rot.per=0.35, 
             colors=brewer.pal(8, "Dark2"))}
dev.off()

##### 3) OUTPUT: PREDICTIONS FOR ALL SUPPORT IDS 

prediction <- data.frame(predict(model_final, dtm, method = "dot")) #dtm could be the one generated at the beginning or another one generated by other dataset
prediction$output <- colnames(prediction)[apply(prediction,1,which.max)]

#prop.table(table(prediction$output)) 
prediction$output %>% table() %>% prop.table() %>% `*`(100) %>% round(2) #ranking of categories (%)

#Manual assign of categories caption 
prediction$output_explained <- prediction$output
prediction$output_explained[prediction$output == "t_1"] <- "Battery Apple"
prediction$output_explained[prediction$output == "t_2"] <- "VirginTrains"
prediction$output_explained[prediction$output == "t_3"] <- "Whatsapp"
prediction$output_explained[prediction$output == "t_4"] <- "Tesco delivery"
prediction$output_explained[prediction$output == "t_5"] <- "Tesco store"
prediction$output_explained[prediction$output == "t_6"] <- "Restart device"
prediction$output_explained[prediction$output == "t_7"] <- "Delete cookies"
prediction$output_explained[prediction$output == "t_8"] <- "British Airways"
prediction$output_explained[prediction$output == "t_9"] <- "Email experience"
prediction$output_explained[prediction$output == "t_10"]<- "Spotify"
prediction$output_explained[prediction$output == "t_11"]<- "Update Apple phone"