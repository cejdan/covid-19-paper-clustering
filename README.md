# CORD-19 Dataset Clustering and Summarizer

Written and developed by Nicholas Cejda for Text Analytics 2020 - Univ. of Oklahoma 26-Apr-2020

The purpose of this project is to utilize the full-text of the Covid-19 related scientific papers published on Kaggle, at https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge

This dataset contains approx. 50,000 scientific papers in JSON format. Our task is to take a subset of these papers, and perform vectorization of the text, and cluster the papers based on their similarities and differences. Then, we are to summarize the text contained within each cluster. The ultimate purpose is to better understand what is contained in this large dataset without having to manually search through each paper.

## Discussion of the data and data format

The data has two main components:
* metadata.csv
* the .json files with full-text information

### metadata.csv
This file is a .csv containing information about the papers contained within the dataset. It includes information like "Title", "Abstract", and "sha", which is a hash code for all papers encoded from pdf files. Our task for this project was to cluster only files from the pdf papers, and we can use the sha code as a key.

### the .json files
The .json files from pdfs are contained within several folders:
* comm_use_subset/pdf_json/
* noncomm_use_subset/pdf_json/
* custom_license/pdf_json/
* biorxiv_medrxiv/pdf_json/

There are other .json files in this dataset, and they are named with their Pubmed central IDs, not sha codes. We ignored these papers for this project.

Within the .json files, they contain lots of information about the content of the paper, including the Full Text. The **json_schema.txt** spells out exactly the format of the .json files. We will take advantage of the "body_text" key, and assemble all the "text" subkeys to generate our full text string.

Blocks 1 through 8 in the Jupyter Notebook are designed to read in all the .json files that we need, and create full-text strings, as well as a full abstract string, which we can then use in future Tokenization / Clustering / Summarization steps.

## Discussion of tokenizer

I have chosen to go with the **nltk** word and sentence tokenizers for this project. Nltk gives you flexibility over how the tokenization occurs, and with scientific papers, there are a number of oddities we need to deal with to get good tokens suitable for clustering. For example, all papers will cite other papers, with some variation on the form "(Jones et al. 2019)" etc. We can remove the et al. tokens by simply extending our stopword list like so:

    stop_words = nltk.corpus.stopwords.words('english')
    extraStopWords = ['et','al', 'al.']
    stop_words.extend(extraStopWords)
    
This removes tokens we know will be found in nearly all papers, which should improve the clustering. In my Notebook, I have also removed non-alphanumeric characters with a regex search:

    myText = re.sub(r'[^a-zA-Z0-9\s]','', myText, re.I)
    
As well as removing single letter tokens like 'b', 'c' (often papers will use phrases like "Fig. 2c"):

    myText = re.sub(r'\b[a-zA-Z]\b', '', myText, re.I)
    
As well as removing numbers "24", "33", "2020", which won't be very characteristic for clustering. I have also removed words starting with a number, to further help eliminate the Fig. 2c problem:

    myText = re.sub(r'\b\d*\b', '', myText)
    myText = re.sub(r'\b\d.*\b', '', myText)
   
Additionally, I have removed words starting with a letter and ending in a number, like antiboody 'H123', as these words don't have a lot of semantic meaning useful for clustering.

    myText = re.sub(r'\b[a-zA-Z]\d*\b', '', myText)
    
We then lowercase the text, and remove leading and trailing whitespace created by our subsitutions, then finally we are ready to tokenize. We do a word tokenization, remove all the stopwords with, and join all the tokens back together (because the Tf-IDF word vectorizer we are about to use needs full strings)

    tokens = nltk.word_tokenize(myText)
    clean_tokens = [t for t in tokens if t not in stop_words]
    clean_text.append(' '.join(clean_tokens))
    
Finally, we then apply a Tf-IDF vecotrizer to our text, to extract features (words) and generate pairwise Tf-IDF scores for each feature for each document. Tf-IDF is a "bag of words" feature extraction method, and weights terms based on their frequency in the documents and in the corpus, high Tf-IDF scores for a particular term in a document indicate that it is an important term in that document. At this point the documents are ready to cluster.
 
 
 ## Discussion of Clustering
 
 Clustering high-dimensionality data like we have involves:
 
 1. Reducing the dimensions so they can be viewed on a 2D or 3D plot.
 2. Assigning cluster numbers to each group of items.
 
 I have selected two diminsionality-reductions approaches:
 
 * T-SNE
 * Latent Semantic Analysis (LSA), also known as Truncated Singular Value Decomposition (SVD)
 
 and one cluster assignment approach:
 
 * K-Means clustering
 
 T-SNE is a popular dimensionality-reduction technique widely used to generate clusters. See https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding for the basics of T-SNE. I applied this algorithm over a range of 'perplexities' (a T-SNE parameter) to obtain the best possible Silhouette Score average (see below).
 
We then applied the truncated SVD method to our data, which is a dimensionality-reducing technique (similar to Principle Component Analysis) but it works on sparse matricies, which is what the output of the Tf-IDF vectorizer is. See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html for more information on truncated SVD. Essentially, it is a linear dimensionality reduction (like PCA), but unlike PCA it is done without centering the data first, allowing it to operate on sparse matricies.

After reducing the dimensions, I checked what the average **Silhouette Score** was for the clusters over a range of K using the K-Means clustering algorithim in order to determine the optimal number of clusters. 

All of these algorithims, T-SNE, truncated SVD, Silhouette Scores, and K-Means are avaliable through the sklearn package.

### Additional notes on Clustering

The first time I attempted to cluster, it was obvious there was some papers sticking way out, detectable by both methods. As it turns out, these were the non-English papers! Our method did a very good job picking up on the language differences. However, I was more interested in clustering based on the *content* of the English papers, so I removed them and tried the clustering again. Unfortunately, there wasn't very much improvement in our clusters. We would up with Silhouette Scores around 0.4. Not terrible but I think with a more sophisticated vectorization method or superior tokenization with Spacy or something better than nltk's basic word vectorizer I could have done better. This dataset is especially difficult to cluster, because the content of the papers is all about Covid-19, or related Coronaviruses, and thus all share many similarities. I attempted clustering on both the Full Text, and the Abstracts (hoping the Abstracts would reveal more distinct clusters), but the two methods wound up with very similar results.


## Discussion of summarization

For this project we used a method of summarization called TextRank. Highly related to the PageRank algorithm developed by Larry Page of Google, this algorithm ranks sentences based on their common elements, and generates a score for how 'well-connected' a sentence is to other sentences. The idea is that the most informative sentences in the cluster will receive the highest TextRank score. We will display the top 5 sentences each cluster as a quick summary of what the most common content is within that cluster. We wrote the cluster summary sentences to the "cluster summary.txt" file. I used the same implementation of TextRank as appears in the textbook:
Text Analytics with Python: A Practitioner's Guide to Natural Language Processing 2nd ed. Edition by Dipanjan Sarkar

### Notes on summarization

So, I attempted to run TextRank on ALL full-text documents for each cluster, and was getting huge matricies around 15,000 x 15,000 elements. I have an old computer, and couldn't feasibly run the program in reasonable time without crashing. To work around this limitation, I decided to calculate the top 10% papers with the smallest Eucledian distance from each cluster's center, and use these central papers as a proxy for the entire cluster's content, figuring the center papers mostly define that cluster. The distance formula was simply:
   
    sqrt( (x1 - y1)^2 + (x2 - y2)^2 ) 
    
This method works to generate cluster summaries in reasonable computational time.

I also tried an alternative method which I think works even better. I simply used the Abstract's sentences in my PageRank instead of the Full Text. This way, I was able to include 100% of the papers within reasonable computational time, and generate actually much more informative summaries. Because a paper's Abstract is already a summary, summarizing the summaries gives more concise descriptions of the content of the cluster. I was much happier with the results from this method, and this is what is included in the "cluster summary.txt" file, although the code to run the Full Text summarization is in the Jupyter Notebook.
