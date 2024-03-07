# Patient-notes-clustering

### Overview

<p>Patient Notes Clustering project is about using the text written by numerous Physicians about 10 different patients. I turned this into an unsupervised machine learning by only considering the physician notes. This natural language processing (NLP) endeavor aims to uncover meaningful patterns and groupings within a dataset of physician notes. By leveraging advanced NLP techniques, we can gain valuable insights into the underlying structure of the data, potentially revealing trends and associations.</p>

### About Dataset

<p>The dataset was given as a part of my NLP coursework. The data has 3 columns out of which two prominent features are `case_num` (indicating patient - has 10 unique values representing 10 different patients) and `pn_history` (notes written by physician).</p>

<p>The `case_num` is used to compare the clustering plot results. On the `pn_history`, we apply Natural language processing techniques and Unsupervised Machine learning techniques to cluster the `pn_history`.</p>

### IDE and Environment

- Programming Language : Python
- IDE : Jupyter Notebooks
- Environment : environment.yml file included

### Data Cleaning and Data preprocessing

<p>There are no missing values in the data. As a part of data preprocessing, I first defined list of contractions and medical abbreviations to be replaced. Later, using regular expressions, I have converted the notes text to lower case, removed number, punctuations and special characters and stop words.</p>
<p>To reduce word to its root form, I compared the results of Lemmatizer and Stemming and decided to use Lemmatization. All these processes are implemented through a function called `preprocess_text`. Later, I have created two new columns, one for original note length and the other for processed text length for visualizations.</p>

### Visualizations

- The following histogram shows us the distribution of note length before and after preprocessing.

<img src="figs/document_length_frequency.png">

Clustering Algorithm
[Describe the clustering algorithm(s) used and why they were chosen.]

Results
[Highlight the main findings and insights gained from the clustering analysis.]

Contributing
[Encourage others to contribute to your project and provide guidelines for how they can do so.]

License
[Specify the license under which your project is distributed.]

Acknowledgements
[Give credit to any tools, libraries, or datasets you used in your project.]

Contact
[Provide your contact information or a way for users to reach out for questions or collaborations.]

Feel free to customize this template based on the specifics of your project. Including clear and detailed information will help others understand and use your natural language processing project effectively.




