# Disaster-Respone-pipeline

### Description
This is project I did as part of my Data-Scientist Nanodegree with udacity.It is a disaster response pipeline which I bulit on the figure eight data-set. It helps disaster recovery team to quickly see which requests are emergency by using text classification. It includes flask and machine learning at back-end.

## Project Overiew

### Project has Two Main pipeline:

#### 1)ETL Pipeline-: 
This is an standard ETL pipeline where data is preprocessed,cleaned,and RAW data is getting transformed to new data which can be used by the machine learning models.

#### 2)ML Pipeline-:
In this pipeline we have used sklearn pipeline to bulit ml pipeline with countvectorizer,tfidftransformer,verb extractor and one estimator(Multioutputclassifier) Adaboostclassifier.

(**Note** -: I tried using random forest but found out by trial and error that Adaboostclassifier has better evaluation score)

### Installation guide

**Step 1**-: Clone the project to local system.

**Step 2** -: Go to the Disaster-Response-Pipeline/app folder using 'cd' command in cmd or terminal.

**Step 3** -: Run 'python run.py'.

**Step 4** -: open 'http://localhost:3001/' in browser.

**step 5** -: **HAVE FUN**

## Author

**Mohit Jawale**



