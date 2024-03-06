#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import Counter
import re
import umap
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


notes = pd.read_csv('data/processed_notes.csv')
notes


# In[3]:


sns.set(style="darkgrid")
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))


sns.histplot(notes['initial_note_length'], bins=50, kde=True, edgecolor='black', ax = axes[0])
axes[0].set_title('Distribution of Document Lengths - before preprocessing')
axes[0].set_xlabel('Document Length')
axes[0].set_ylabel('Frequency')

sns.histplot(notes['processed_note_length'], bins=50, kde=True, edgecolor='black', ax = axes[1])
axes[1].set_title('Distribution of Document Lengths - after preprocessing')
axes[1].set_xlabel('Document Length')
axes[1].set_ylabel('Frequency')

plt.savefig('figs/document_length_frequency.png')
plt.show()


# In[4]:


notes[['initial_note_length', 'processed_note_length']].describe()


# In[5]:


def get_common_words(texts, num_words=10):

    words = texts.split()
    word_counter = Counter(words)

    # Get the most common words and their frequencies
    common_words = word_counter.most_common(num_words)

    return common_words


data = notes[['case_num','processed_notes']].groupby(['case_num']).agg(lambda x: ' '.join(x))

sns.set(style="darkgrid")
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(30, 50))

for i,j in data['processed_notes'].apply(lambda x: get_common_words(x, num_words = 20)).items():
    #print(f"patient case number - {i}.\nMost common words in those notes (combined):")
    case = pd.DataFrame(data = j, columns = ['word', 'count'])
    
    plot_row = i // 2
    plot_col = i % 2
    
    axes[plot_row][plot_col].bar(case['word'],case['count'])
    axes[plot_row][plot_col].tick_params(labelrotation=45)
    axes[plot_row][plot_col].set_title(f'Most common words in Patient Case - {i} Notes (combined)')

plt.savefig('figs/common_words_per_patient_case.png')
plt.show()


# In[6]:


text_combined = ' '.join(data['processed_notes'])

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(text_combined)

# Display the word cloud using Matplotlib
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Turn off axis labels

plt.savefig('figs/word_cloud_all_notes.png')
plt.show()


# In[7]:


df = pd.DataFrame(data = get_common_words(text_combined, num_words = 20), columns = ['word', 'count'])
plt.figure(figsize = (20,15))
plt.bar(df['word'], df['count'])
#plt.xticks(rotation = 60)
plt.title('Common words - in all texts combined')
plt.savefig('figs/common_words_all_notes.png')
plt.show()


# ### Notes:
# 
# - We can see that there are 42146 documents but there are almost 10 words which occur more than 42146 times - meaning they are present in almost all documents.
# 
# - We can observe that almost all of the top 20 words are common are occur in most of the documents and might not be as important. We can filter these words out by setting max_df for TFIDF.
