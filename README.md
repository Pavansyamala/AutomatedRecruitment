# Infosys Project: Interview Automation

## Project Overview
This project focuses on automating the interview process by predicting whether a candidate is selected or rejected based on the interview transcript. The project aims to leverage natural language processing (NLP) techniques and machine learning models to streamline and optimize the interview process, reducing manual effort and improving decision-making.

## Objectives
1. Automate the process of evaluating candidates based on their interview transcripts.
2. Use exploratory data analysis (EDA) to uncover meaningful patterns and insights from the dataset.
3. Build a robust predictive model for candidate selection.

## Dataset Overview
The dataset includes:
- **Interview Transcripts**: Text data capturing the interaction between interviewers and candidates.
- **Labels**: Binary outcomes (`Selected` or `Rejected`) for each candidate.
- **Additional Metadata**: Information such as candidate ID, job description, and interviewer ID.

## Work Completed

### 1. Data Cleaning and Preprocessing
- Removed null and duplicate records.
- Normalized text data (lowercased, removed stopwords, punctuation, and special characters).
- Tokenized interview transcripts and applied lemmatization for semantic consistency.

### 2. Exploratory Data Analysis (EDA)
#### Key Statistics:
1. **Statistical Insights**:
   - A T-Test revealed that the number of words in the transcript is significantly different for selected and rejected candidates.
2. **Similarity Analysis with TF-IDF**:
   - The similarity between Transcript and Job Description is higher for selected candidates.
   - The similarity between Transcript and Resume and Resume and Job Description is comparatively lower.
3. **Similarity Analysis with Word2Vec**:
   - The similarity between Resume and Job Description, Job Description and Transcript, and Transcript and Resume is higher for selected candidates compared to rejected candidates.
4. **Role-Based Data Filtering**:
   - Role-specific filtering of data helped identify key patterns for candidates matching job-specific requirements.

#### Other Statistics:
- **Transcript Length**:
  - Average length of transcripts: **250 words**
  - Minimum length: **50 words**, Maximum length: **500 words**
- **Word Frequency**:
  - Top 5 most frequent words: `experience`, `skills`, `project`, `team`, `development`.

#### Insights:
- Candidates with concise and to-the-point answers had higher chances of selection.
- Use of technical keywords in transcripts was positively correlated with selection.
- Candidates who mentioned teamwork and leadership more frequently had better outcomes.

### 3. Feature Engineering
- Extracted **TF-IDF features** from interview transcripts.
- Performed **sentiment analysis** using VADER to include sentiment polarity as a feature.
- Incorporated **word embeddings** (Word2Vec/Glove) to capture semantic meaning.

### 4. Model Development
- Built initial models using:
  - Logistic Regression
  - Random Forest Classifier
  - Support Vector Machine (SVM)
- Evaluated models using **accuracy**, **precision**, **recall**, and **F1-score**.

## Next Steps
1. Fine-tune models using hyperparameter optimization.
2. Experiment with advanced deep learning models (e.g., BERT, GPT) for better transcript understanding.
3. Develop a user-friendly interface for HR teams to input transcripts and view predictions.

## Tools and Technologies Used
- **Programming Language**: Python
- **Libraries**: pandas, numpy, scikit-learn, nltk, spacy, tensorflow
- **Visualization Tools**: matplotlib, seaborn
- **Environment**: Jupyter Notebook

## Contributors
- **Kasa Pavan**

## Acknowledgements
This project is part of an ongoing collaboration with Infosys to optimize and automate HR processes.
