#!/usr/bin/env python
# coding: utf-8

# In[97]:


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import contractions
import numpy as np


# ## Reading the Patient notes

# In[98]:


#reading the patient notes data

file_path = "data/patient_notes.csv"
notes = pd.read_csv(file_path)
notes


# ### Missing value report

# In[99]:


#missing value report
notes.isna().sum()


# ### Preprocessing

# In[100]:


print(f"There are {len(notes['case_num'].unique())} patients and {notes.shape[0]} patient notes in total.")
print(f"\nNumber of patient notes per case:\n{notes['case_num'].value_counts()}")


# In[101]:


#setting up few additional contractions to preprocess the data

contractions.add('yo', 'year old')
contractions.add('y o', 'year old')
contractions.add('y.o.', 'year old')
contractions.add('y.o', 'year old')
contractions.add('mos', 'months')
contractions.add('mo', 'months')
contractions.add('min', 'minutes')
contractions.add('mins', 'minutes')
contractions.add('y/o', 'year old') 
contractions.add('yr,', 'year')
contractions.add('m', 'male')
contractions.add('f','female')
contractions.add('yr', 'year')


# In[102]:


#taken from here, slightly edited: https://www.dukemychart.org/home/en-us/docs/commonmedicalabbreviations.pdf
# adding more medical abbrevations to contractions list

test = '''A/P: Assessment and Plan
BMI: Body Mass Index
BMP: Basic Metabolic Profile 
BP: Blood Pressure
C&S: Culture and Sensitivity
C/O: complains of
CBC: Complete Blood Count
CC: Chief Complaint
CCE: clubbing, cyanosis or edema
Chemistry: a blood test looking at levels of electrolytes and kidney or liver function
Chem Panel: a blood test looking at levels of electrolytes and kidney or liver function
CKD: Chronic Kidney Disease
CMP: a blood test looking at levels of electrolytes, kidney and liver function
D/Dx: Differential Diagnosis
DOE: Dyspnea on exertion
DM: Diabetes Mellitus
DMII: Diabetes Mellitus Type II
ECG/EKG: Electrocardiogram
EOMI: Extra-ocular eye movements intact
ESRD: End Stage Renal Disease
ETOH: Alcohol
ETT: Endotracheal tube 
EXT: Extremities
F/U: Follow-up
GI: Gastrointestinal
GU: Genito-urinary (referring to the Urinary Tract)
H&H: Hemoglobin and Hematocrit
H&P: History and Physical
HCT: Hematocrit
HGB: Hemoglobin
HgBA1C: A blood test that measures your average blood glucose control over the last 3 months
HPI: History of the Present Illness
HEENT: Head, Ears, Eyes, Nose and Throat
HTN: Hypertension (High Blood Pressure)
I&D: Incision and Drainage
IM: intra-muscular
IMP: Impression
IV: Intra-venous
LBP: low back pain
LMP: last menstrual period
ND: naso-duodenal 
Neuro: Neurologic 
NG: naso-gastric
NJ: naso-jejunal
N/V: nausea and vomiting
OT: Occupational Therapy 
P: pulse
PCP: Primary Care Provider
PERRLA: Pupils equal, round and reactive to light and accommodation
PLT: Platelets
PMHx:Past Medical History
PO: to be taken by mouth
PR: to be taken by rectum
PRN: As needed
PSHx: Past Surgical History
Pt: patient
Renal Function Panel: a blood test looking at levels of electrolytes and kidney function
R/O: Rule Out
RR: Respiratory Rate
SocHx or SH:Social History
SOB: Shortness of breath
SQ: Sub-cutaneous
ST: Speech Therapy
STI: Sexually transmitted infection
T: Temperature
TM: Tympanic membrane
UA: Urinalysis
URI: Upper Respiratory Infection
UTI: Urinary Tract Infection
VSS: Vital Signs Stable
WBC: White blood cell 
WCC: Well Child Check
WT: Weight
PMH: Past Medical History'''

for x in test.split('\n'):
    contractions.add(x.split(':')[0], x.split(':')[1])


# In[103]:


def preprocess_text(text, flag):
    '''
    preprocessing the required text column to convert case, remove number, remove contractions and stopwords
    '''
    # Convert to lower case
    text = text.lower()
    
    ## add space inbetween numbers and letters (e.g. 5mg to 5 mg, 17yo to 17 yo)
    text = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', text)
    
    #remove numbers
    text = re.sub(r'\d+', '', text)

    # Expand contractions (e.g., "can't" to "can not")
    text = contractions.fix(text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    #removing stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    # Using Lemmatizer or Stemmer
    if flag:
        lemmatizer = WordNetLemmatizer()
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    else:
        stemmer = PorterStemmer()
        text = ' '.join([stemmer.stem(word) for word in text.split()])

    return text


# In[104]:


# Apply the preprocessing to the 'pn_history' column - stemming the text
notes['stemmed_pn_history'] = notes['pn_history'].apply(lambda x: preprocess_text(x, False))
for i in notes['stemmed_pn_history'].head(3):
    print(i)
    print()


# In[105]:


# Apply the preprocessing to the 'pn_history' column - lemmatizing the text
notes['lemmatize_pn_history'] = notes['pn_history'].apply(lambda x: preprocess_text(x, True))
for i in notes['lemmatize_pn_history'].head(3):
    print(i)
    print()


# <br>
# <br>
# 
# ### Notes:
# 
# - Comparing the processed text from both lemmatized and stemmed text, using lemmatizer makes more sense.
# 
# - Therefore, I am using Lemmatization.

# In[106]:


#creating initial note length column
notes['initial_note_length'] = notes['pn_history'].apply(lambda x: len(x))


# In[107]:


#dropping other unnecessary columns along with "stemmed_pn_history"
notes.drop(['pn_num', 'pn_history', 'stemmed_pn_history'], axis=1, inplace = True)

#renaming lemmatized column name
notes.rename({'lemmatize_pn_history' : 'processed_notes'}, inplace=True, axis = 1)

#creating processed note length column
notes['processed_note_length'] = notes['processed_notes'].apply(lambda x: len(x))
notes


# In[108]:


notes.to_csv('data/processed_notes.csv', index = False)

