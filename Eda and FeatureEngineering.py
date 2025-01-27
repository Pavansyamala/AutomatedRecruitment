#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import string 
import seaborn as sns 
import nltk 
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os 
import contractions 


# ## Findings from Exploratory Data Analysis 
# 
# #### 1. Statistically its proven that The number of words in Transcript is different for Selection and Rejection Candidates (Using T-Test) 
# 
# #### 2. Using the Tf-IDF vectorizer , it cames out that only similarity between transcript and Job decription is higher for selected candidates compared with similarity between transcript and resume , resume and Job description 
# 
# #### 3. But when we used the Word to Vector model, the similarity between Resume and Job description, Job description and Transcript , Transcript and Resume are higher for selected candidates compared with the rejected candidates
# 
# #### 4. data analyst,data engineer,data scientist,product manager,software engineer,ui designer,ui engineer  are the only roles which have only enough data to train on all other roles have data between 10-60 data points, so we dropped them 
# 

# In[61]:


path = "D:/InterviewAutomation/data"


# In[63]:


excel_files = [file for file in os.listdir(path) if file.endswith(('.xls', '.xlsx'))]


# In[64]:


for file in excel_files:
    df = pd.read_excel(path+'/'+file)
    print(df.columns)


# In[5]:


# # Changing the Columns of the 4th dataset as it does have the Wrong Columns
# df = pd.read_excel(excel_files[3])
# df.columns = ['ID', 'Name', 'Role', 'Transcript', 'Resume', 'decision',
#        'Reason for decision', 'Job Description']
# df.to_excel(path+'/'+'dataset4.xlsx',index=False)


# In[6]:


# for file in excel_files:
#     df = pd.read_excel(path+'/'+file)
#     print(df.columns)


# In[65]:


# Code for Merging all the datasets and creating the full dataset for traing and testing
dataframes = []

for file in excel_files:
    file_path = os.path.join(path, file)
    df = pd.read_excel(file_path)
    dataframes.append(df)

# Combine all dataframes into a single dataframe
data = pd.concat(dataframes, ignore_index=True)


# In[66]:


data.shape


# In[67]:


data.head()


# In[68]:


data.info()


# In[69]:


for col in data.columns:
    data[col] = data[col].apply(lambda x: x.lower())


# In[70]:


data['decision'].unique()


# In[71]:


def process_decision(text):
    if text in ['select','selected']:
        return 'select'
    else :
        return 'reject'


# In[72]:


data['decision'] = data['decision'].apply(process_decision )


# In[73]:


data['Role'].unique()


# In[74]:


unique_count = data.groupby('Role')['ID'].count()
unique_count


# In[75]:


## Dropping Rows which have count < 100 
for count , role in zip(unique_count.values, unique_count.index):
    if count < 100 :
        data = data[data['Role'] != role] 


# In[76]:


unique_count = data.groupby('Role')['ID'].count()
unique_count


# In[77]:


data.shape


# In[78]:


## Performing Feature Engineering
string.punctuation


# In[81]:


nlp = spacy.load("en_core_web_sm") 
stopword_s = stopwords.words('english')
lemmatizer = WordNetLemmatizer() 

def textpreprocessing(text):
    text = contractions.fix(text) # expanding the contractions like , I'm with I am and it's with It is and so on...
    text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\]^_`{|}~]', '', text)  # Removing all these characters from the text 
    sentences = re.split(r'(?<=[.!?])\s+', text)   # Splitting the text as sentences using the regular expressions     
    for idx, sent in enumerate(sentences):
        words = nlp(sent)  # converting the sentence into words
        words = [word.text for word in words if word.text.lower() not in stopword_s] # Removing the stopwords from the text
        words = [lemmatizer.lemmatize(word) for word in words]  # Using the Lemmatization techniques
        sentences[idx] = ' '.join(words)  # Replacing the original sentence with the new sentence 
    return ' '.join(sentences)  # Join the sentences back together  


# In[83]:


data['Transcript_processed'] = data['Transcript'].apply(textpreprocessing)
# Saving the processed Data to a temporary file
data.to_excel(path+'/'+'final_data.xlsx')


# In[84]:


data['Resume_processed'] = data['Resume'].apply(textpreprocessing)
# Saving the processed Data to a temporary file
data.to_excel(path+'/'+'final_data.xlsx')


# In[85]:


data['Job_Description_processed'] = data['Job Description'].apply(textpreprocessing)
# Saving the processed Data to a temporary file
data.to_excel(path+'/'+'final_data.xlsx')


# In[2]:


path_final = "D:/InterviewAutomation/data/final_data.xlsx"


# In[3]:


df = pd.read_excel(path_final)


# In[ ]:


df.head()


# In[5]:


new_df = df.drop(columns=['Unnamed: 0', 'Transcript' , 'Resume' , 'Job Description'])


# In[6]:


new_df


# In[7]:


new_df['Transcript_words'] = new_df['Transcript_processed'].apply(lambda x: len(x.split())) 
new_df['Transcript_length'] = new_df['Transcript_processed'].apply(lambda x:len(x))

new_df['Resume_words'] = new_df['Resume_processed'].apply(lambda x: len(x.split())) 
new_df['Resume_length'] = new_df['Resume_processed'].apply(lambda x:len(x))


# In[8]:


new_df.head()


# In[9]:


avg_words_role = new_df.groupby(['Role','decision'])['Transcript_words'].mean()


# In[10]:


avg_words_role


# In[12]:


## Statistical Analysis for checking wether there is a significant difference between the number of words in Transcript, for select and reject


# #### Null Hypothesis : The mean number of words in transcript for the decision of rejection and selection are equal 
# #### Alternative hypothesis : The mean number of words in transcript for the decision of rejection and selection are not equal

# In[11]:


new_df.groupby(['decision'])['Transcript_words'].mean()


# In[12]:


means = new_df.groupby(['decision'])['Transcript_words'].mean().values


# In[13]:


reject_sample = new_df[new_df['decision'] == 'reject'].sample(300)
select_sample = new_df[new_df['decision'] == 'select'].sample(300)


# In[14]:


reject_var = (reject_sample['Transcript_words'].std())**2 
select_var = (select_sample['Transcript_words'].std())**2


# In[15]:


reject_var/select_var  # This is in between 0.5 and 2, so we need to take the variances as almost equal 


# In[16]:


# Using T-distribution for testing 
# Sampling the original distribution 

from scipy.stats import ttest_ind

# Perform a two-sample t-test


t_stat, p_value = ttest_ind(
    reject_sample['Transcript_words'], 
    select_sample['Transcript_words'], 
    equal_var=False # Use Welch's t-test if variances are unequal
)

print("T-statistic:", t_stat)
print("P-value:", p_value)


# In[17]:


## From this we get to know that , there is a significant evidence to reject Null hypothesis that the average number of words for select is equal to the rejection


# In[18]:


def T_test(role , reject_sample,select_sample):
    t_stat, p_value = ttest_ind(
        reject_sample['Transcript_words'], 
        select_sample['Transcript_words'], 
        equal_var=False  # Use Welch's t-test if variances are unequal
    )

    if p_value < 0.05 :
        print("Rejecting Null Hypothesis , Mean number of Transcript words for rejection and selection are significantly different for the role of {}".format(role))
    else : 
        print("Accepting Null Hypothesis , Mean number of Transcript words for rejection and selection are equal for the role of {}".format(role))


# In[19]:


def role_wise_test(roles):
    for role in roles:
        curr_df = new_df[new_df['Role'] == role] 
        reject_sample = curr_df[curr_df['decision'] == 'reject'].sample(100)
        select_sample = curr_df[curr_df['decision'] == 'select'].sample(100) 
        T_test(role , reject_sample , select_sample)


# In[20]:


role_wise_test(new_df['Role'].unique())


# In[21]:


## From this we know that, for every role there is a significant difference between the average number of words for selection and rejection for every role


# In[22]:


new_df.groupby(['Role'])['Resume_words'].mean() # From this the average number of words in resume for every role are almost equal


# In[23]:


def T_test1(role , reject_sample,select_sample):
    t_stat, p_value = ttest_ind(
        reject_sample['Transcript_length'], 
        select_sample['Transcript_length'], 
        equal_var=False  # Use Welch's t-test if variances are unequal
    )

    if p_value < 0.05 :
        print("Rejecting Null Hypothesis , Mean length of Transcript for rejection and selection are significantly different for the role of {}".format(role))
    else : 
        print("Accepting Null Hypothesis , Mean length of Transcript for rejection and selection are equal for the role of {}".format(role))


# In[24]:


new_df.groupby(['Role','decision'])['Transcript_length'].mean()


# In[25]:


def role_wise_test1(roles):
    for role in roles:
        curr_df = new_df[new_df['Role'] == role] 
        reject_sample = curr_df[curr_df['decision'] == 'reject'].sample(100)
        select_sample = curr_df[curr_df['decision'] == 'select'].sample(100) 
        T_test1(role , reject_sample , select_sample)


# In[26]:


role_wise_test1(new_df['Role'].unique())


# In[28]:


## Finding wether there is a similarity between the two transcripts if so exists, we will drop one of them


# In[27]:


## Finding wether there is a similarity between the two transcripts if so exists and we will drop the other transcripts 
## For 90% similarity
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(new_df['Transcript_processed'])

cosine_sim = cosine_similarity(tfidf_matrix)

# Find transcript pairs with similarity >= 0.9
threshold = 0.9
similar_dict = {}

for i in range(cosine_sim.shape[0]):
    similar_indices = []
    for j in range(i + 1, cosine_sim.shape[1]):  # Avoid redundant comparisons
        if cosine_sim[i, j] >= threshold:
            similar_indices.append(j)
    if similar_indices:
        similar_dict[i] = similar_indices

# Print the dictionary
print("Similar Transcripts Dictionary (≥ 90% similarity):")
print(similar_dict)


# In[28]:


## Finding wether there is a similarity between the two transcripts if so exists and we will drop the other transcripts 
## For 80% similarity
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(new_df['Transcript_processed'])

cosine_sim = cosine_similarity(tfidf_matrix)

# Find transcript pairs with similarity >= 0.8
threshold = 0.8
similar_dict = {}

for i in range(cosine_sim.shape[0]):
    similar_indices = []
    for j in range(i + 1, cosine_sim.shape[1]):  # Avoid redundant comparisons
        if cosine_sim[i, j] >= threshold:
            similar_indices.append(j)
    if similar_indices:
        similar_dict[i] = similar_indices

# Print the dictionary
print("Similar Transcripts Dictionary (≥ 80% similarity):")
print(similar_dict)


# In[151]:





# In[29]:


tfidf_vectorizer1 = TfidfVectorizer()
transcript_matrix = tfidf_vectorizer1.fit_transform(new_df['Transcript_processed'])

tfidf_vectorizer2 = TfidfVectorizer()
resume_matrix = tfidf_vectorizer2.fit_transform(new_df['Resume_processed'])

tfidf_vectorizer3 = TfidfVectorizer()
jobDescription_matrix = tfidf_vectorizer3.fit_transform(new_df['Job_Description_processed'])


# In[30]:


# Use a single TfidfVectorizer for both Resume and Job Description
tfidf_vectorizer4 = TfidfVectorizer()
combined_text = new_df['Resume_processed'] + " " + new_df['Job_Description_processed']
tfidf_vectorizer4.fit(combined_text)

# Split the combined TF-IDF matrix back into Resume and Job Description matrices
resume_matrix = tfidf_vectorizer4.transform(new_df['Resume_processed'])
jobDescription_matrix = tfidf_vectorizer4.transform(new_df['Job_Description_processed'])

# Calculate row-wise cosine similarity
cos_rjd = []
for i in range(new_df.shape[0]):
    cos_rjd.append(cosine_similarity(resume_matrix[i], jobDescription_matrix[i])[0][0])


# In[31]:


new_df['Resume_JD'] = cos_rjd


# In[32]:


# Use a single TfidfVectorizer for both Resume and Transcript
tfidf_vectorizer5 = TfidfVectorizer()
combined_text = new_df['Resume_processed'] + " " + new_df['Transcript_processed']
tfidf_vectorizer5.fit(combined_text)

# Split the combined TF-IDF matrix back into Resume and Job Description matrices
resume_matrix = tfidf_vectorizer5.transform(new_df['Resume_processed'])
Transcript_matrix = tfidf_vectorizer5.transform(new_df['Transcript_processed'])

# Calculate row-wise cosine similarity
cos_rt = []
for i in range(new_df.shape[0]):
    cos_rt.append(cosine_similarity(resume_matrix[i], Transcript_matrix[i])[0][0])


# In[33]:


new_df['Resume_Transcript'] = cos_rt


# In[34]:


# Use a single TfidfVectorizer for both Job description and Transcript
tfidf_vectorizer6 = TfidfVectorizer()
combined_text = new_df['Job_Description_processed'] + " " + new_df['Transcript_processed']
tfidf_vectorizer6.fit(combined_text)

# Split the combined TF-IDF matrix back into Resume and Job Description matrices
JD_matrix = tfidf_vectorizer6.transform(new_df['Job_Description_processed'])
Transcript_matrix = tfidf_vectorizer6.transform(new_df['Transcript_processed'])

# Calculate row-wise cosine similarity
cos_jdt = []
for i in range(new_df.shape[0]):
    cos_jdt.append(cosine_similarity(JD_matrix[i], Transcript_matrix[i])[0][0])


# In[35]:


new_df['JobDescription_Transcript'] = cos_jdt


# In[36]:


new_df.head()


# In[37]:


new_df.groupby(['decision'])['Resume_Transcript'].mean()


# In[38]:


new_df.groupby(['decision'])['Resume_JD'].mean()


# In[39]:


new_df.groupby(['decision'])['JobDescription_Transcript'].mean()


# In[40]:


import gensim
from gensim.models import Word2Vec

# Preprocess the transcripts
def preprocess_text(text):
    # Tokenize text
    tokens = text.split()
    return tokens

# Preprocess all transcripts
processed_transcripts = [preprocess_text(transcript) for transcript in new_df['Transcript_processed']]


# In[41]:


# Train Word2Vec model
model = Word2Vec(sentences=processed_transcripts, vector_size=100, window=5, min_count=1, workers=4)

# Save the model for future use



# In[42]:


model.save("word2vec_transcript_model")


# In[43]:


# Example: Vectorizing a single transcript
def vectorize_transcript(transcript, model):
    tokens = preprocess_text(transcript)
    # Get the word vectors for each word in the transcript and average them
    word_vectors = [model.wv[word] for word in tokens if word in model.wv]
    if word_vectors:
        return sum(word_vectors) / len(word_vectors)
    else:
        return None  # If no words found in the model


# Example: You can also vectorize all transcripts at once
Transcript_vectors = np.array([vectorize_transcript(transcript, model) for transcript in new_df['Transcript_processed']])


# In[44]:


processed_resumes =  [preprocess_text(resume) for resume in new_df['Resume_processed']]
processed_JD =  [preprocess_text(jd) for jd in new_df['Job_Description_processed']]


# In[45]:


model1 =  Word2Vec(sentences=processed_resumes, vector_size=100, window=5, min_count=1, workers=4)
model2 = Word2Vec(sentences=processed_JD, vector_size=100, window=5, min_count=1,workers=4)


# In[46]:


Resume_vectors = np.array([vectorize_transcript(resume, model1) for resume in new_df['Resume_processed']])
JD_vectors = np.array([vectorize_transcript(jd, model2) for jd in new_df['Job_Description_processed']])


# In[58]:


model1.save("word2vec_resume_model")
model2.save("word2vec_job_description_model")


# In[47]:


re_tr_simi_w2v = [] 
re_jd_simi_w2v = []
tr_jd_simi_w2v = [] 

for i in range(new_df.shape[0]):
    re_tr_simi_w2v.append(cosine_similarity(Resume_vectors[i].reshape(-1,1),Transcript_vectors[i].reshape(-1,1))[0][0])
    re_jd_simi_w2v.append(cosine_similarity(Resume_vectors[i].reshape(-1,1),JD_vectors[i].reshape(-1,1))[0][0]) 
    tr_jd_simi_w2v.append(cosine_similarity(Transcript_vectors[i].reshape(-1,1),JD_vectors[i].reshape(-1,1))[0][0]) 


# In[48]:


new_df['Resume_Transcript_W2V'] = re_tr_simi_w2v 
new_df['Resume_JD_W2V'] = re_jd_simi_w2v 
new_df['JobDescription_Transcript_W2V'] = tr_jd_simi_w2v 


# In[49]:


new_df.groupby(['decision'])['Resume_Transcript_W2V'].mean()


# In[50]:


new_df.groupby(['decision'])['Resume_JD_W2V'].mean()


# In[51]:


new_df.groupby(['decision'])['JobDescription_Transcript_W2V'].mean()

