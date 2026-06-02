**Milestone 4: Final Submission**

**Introduction**

Online shopping platforms contain millions of customer reviews, but only a small fraction are considered helpful by other users. Identifying helpful reviews can automatically improve product discovey and support better purchasing decisions. In this project, we used distributed machine learning models on a large Amazon Reviews dataset to predict whether a review will receive helpful votes. 
Since the data contains more than 58 million reviews, traditional single-machine processing is impractical. Therefore, Apaache Spark was deployed on SDSC Expanse to perform scalable data processing, feature engineering, and model training. The project explores both structured review features and natural language processing techniques to determine which factors contribute most to review helpfulness. 


**SDSC Expanse Environment**
All experiments were conducted on SDSC Expanse  using Jupyter Notebook with PySpark. The Spark environment was configured with 64 GB of memory and 8 CPU cores. One core was reserved for the Spark driver, resulting in seven executors with approximately 8 GB of memory each. This configuration enabled distributed processing of the large-scale dataset. 

**Data Exploration**
The Amazon Reviews dataset contained 58,075,027 observations and 15 variables. Exploratory analysis revealed that:
- Star ratings were heavilt concentrated in the 4-5 star range.
- Helpful votes and total votes were highly right-skewed.
- Product categories varies substantially in review volume.
- Technology-related categories such as Wireless and PC products received the largest number of reviews.


**Preprocessing**
Data preprocessing was performed using Spark DataFram operations. 
The following steps were applied: 
- Removed records with missing values in key columns.
- Converted rating and voting variables to numeric format.
- Generated a review length feature from the review text.
- Created a binary target variable (helpful_label) where reviews receiving at least one helpful vote were labeles as helpful.
- Applied tokenization and stop-word removal to review text.
- Converted text into numerical feature vectors using TF-IDF
- Reduced dimensionality using PCA with 80 principal components

To reduce computational cost while maintaining dataset characteristics, a 1% sample of the dataset was used for model training and evaluation. 

**Model 1: Decision Tree Classification**
The first distributed model used a DT classifier trained on PAC-transformed TF-IDF features. 
Model parameters:
- Maximum Depth: 5
- Training/Test split: 80/20
- Features: PCA-transformed review text
Model performance was evaluated using classification accuracy, confusion matrices, and PCA variance analysis.

**Model 2: Random Forest Classification**
The second distributed model used a RF classifier with the same preprocessing pipeline.
Model parameters:
- Number of trees: 5
- Training/Test split: 80/20
- Features: PCA-transformed review text
Model performance was evaluated using classification accuracy, confusion matrices, and PCA variance analysis.

**Model                  Test Accuracy**
Decision Tree             74.76%
Random Forest             74.19%

The detailed results, methods, and discussion are presented in the accompanying written report. 


**Fitting Analysis:**

• For our second model we receives a training accuracy of ___ and a test accuracy of ___. The two accuracy are in close proximity of each other overfitting doesn't seem to be a huge issue. The model is doing a decent job of learning pattern about the data.

• Potential future improvements: to improve our model we could increase our PCA from 50 to 100 and hashing up to 2500-5000. By increasing the hashing we allow for wider variety of word patterns which the PCA can then compress into a valuable set of words, after the idf is applied. We could also potentially switch to using boosted trees to pick up complex patterns that weren't identified in the previous model.

• Dimensionality Reduction: Dimensionality reduction helped us gather insights on the review text but only on text that was deemed important. If we tried to test on every word in the review text the code would've probably crashed. Using a PCA-based model allowed us to pick the best words that had the most accuracy, which made modeling a lot easier.


# Data URL & Code
- **Dataset:** https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset
- **Notebook:** https://github.com/Zeina08/Amazon-US-Customer-Reviews-Dataset/blob/main/milestones/ReviewDataset.ipynb

# Environment:
- **Platform:** SDSC Expanse (ACCESS CI allocation portal)
- **Framework:** PySpark
