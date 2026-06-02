**Milestone 4: Final Submission**



**Introduction**

Online shopping platforms contain millions of customer reviews, but only a small fraction are considered helpful by other users. Identifying helpful reviews can automatically improve product discovery and support better purchasing decisions. 
In this project, we used distributed machine learning models on a large Amazon Reviews dataset to predict whether a review will receive helpful votes. 
Since the data contains more than 58 million reviews, traditional single-machine processing is impractical. Therefore, Apache Spark was deployed on SDSC Expanse to perform scalable data processing, feature engineering, and model training. The project applies natural language processing techniques and machine learning methods to predict review helpfulness from review text. 


**SDSC Expanse Environment** 🛠️

All experiments were conducted on SDSC Expanse  using Jupyter Notebook with PySpark. The Spark environment was configured with 64 GB of memory and 8 CPU cores. One core was reserved for the Spark driver, resulting in seven executors with approximately 8 GB of memory each. This configuration enabled distributed processing of the large-scale dataset. 

**Data Exploration** 🔍

The Amazon Reviews dataset contained 58,075,027 observations and 15 variables. Exploratory analysis revealed that:
- Star ratings were heavilt concentrated in the 4-5 star range.
- Helpful votes and total votes were highly right-skewed.
- Product categories varied substantially in review volume.
- Technology-related categories such as Wireless and PC products received the largest number of reviews.



**Preprocessing** 💻

Data preprocessing was performed using Spark DataFrame operations. 
The following steps were applied: 
- Removed records with missing values in key columns.
- Converted rating and voting variables to numeric format.
- Generated a review length feature from the review text.
- Created a binary target variable (helpful_label) where reviews receiving at least one helpful vote were labeled as helpful.
- Applied tokenization and stop-word removal to review text.
- Converted text into numerical feature vectors using TF-IDF.
- Reduced dimensionality using PCA with 80 principal components.

To reduce computational cost while maintaining dataset characteristics, a 1% sample of the dataset was used for model training and evaluation. 

**Model 1: Decision Tree Classification**

The first distributed model used a DT classifier trained on PCA-transformed TF-IDF features. 
Model parameters:
- Maximum Depth: 5
- Training/Test split: 80/20
- Features: PCA-transformed review text

-> Model performance was evaluated using classification accuracy, confusion matrices, and PCA variance analysis.

**Model 2: Random Forest Classification**

The second distributed model used a RF classifier with the same preprocessing pipeline.
Model parameters:
- Number of trees: 20
- Training/Test split: 80/20
- Features: PCA-transformed review text

-> Model performance was evaluated using classification accuracy, confusion matrices, and PCA variance analysis.



**Fitting Analysis:** 📊📈

• For our second model we receives a training accuracy of ___ and a test accuracy of ___. The two accuracy are in close proximity of each other overfitting doesn't seem to be a huge issue. The model is doing a decent job of learning pattern about the data.

• Potential future improvements: to improve our model we could increase our PCA from 80 to 100 and hashing up to 2500-5000. By increasing the hashing we allow for wider variety of word patterns which the PCA can then compress into a valuable set of words, after the idf is applied. We could also potentially switch to using boosted trees to pick up complex patterns that weren't identified in the previous model.

• Dimensionality Reduction: Dimensionality reduction helped us gather insights on the review text but only on text that was deemed important. If we tried to test on every word in the review text the code would've probably crashed. Using a PCA reduced dimenstionality of the TF-IDF feature space by compressing inforamation from many text features into a smaller set of principal components. This reduced computational cost while preserving a significant portion of the original variance. 


# Data URL & Code 🔗
- **Dataset:** https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset
- **Notebook:** https://github.com/Zeina08/Amazon-US-Customer-Reviews-Dataset/blob/main/milestones/ReviewDataset.ipynb

# Environment:
- **Platform:** SDSC Expanse (ACCESS CI allocation portal)
- **Framework:** PySpark







**Written Report Sections**

**1.	Introduction**

This project focuses on the prediction of helpful Amazon product reviews using machine learning models. With the large volume of user-generated content available on e-commerce platforms, identifying which reviews are likely to be useful can improve the browsing experience and support decision-making for customers. 
Two machine learning models (Decision Tree and Random Forest) are implemented for review helpfulness classification. Both models use a unified text-based feature pipeline consisting of tokenization, stop-word removal, TF-IDF feature extraction, and PAC for dimensionality reduction.
Decision Tree and Random Forest classifiers are applied to both feature representations in order to compare their performance. The goal is to evaluate how different feature engineering strategies and classification models influence the ability to accurately predict review helpfulness. 



**2.	Figures**

Figure A. Confusion Matrix of the Decision Tree.
The below matrix summarizes the model’s classification performance by comparing predicted and actual review labels. Most non-helpful reviews were correctly classified while a larger number of helpful reviews were misclassified as non-helpful, which indicates a lower recall for the helpful class. 


Figure B. Confusion Matrix of the Random Forest.
The below matrix illustrated the classification results by comparing predicted and actual review labels. The model correctly identified 77,059 non-helpful reviews and 9,159 helpful reviews, while misclassifying a substantial number of helpful reviews as non-helpful, indicating stronger performance on the majority class than on the helpful class. 

Figure C. Distribution of Helpful and Not Helpful Reviews.
The below bar chart shows the class distribution in the dataset. The predominance of non-helpful reviews indicates a class imbalance, which may explain the lower recall observed for the helpful class in both Decision Tree and Random Forest models.  


Figure D. Average review length by helpfulness class. 
The helpful reviews tend to be longer than the non-helpful reviews, which indicates that review length or more detailed reviews may be associated with perceived usefulness. 

Figure E. Comparison of test accuracy between the Decision Tree and Random Forest models.
Both models achieved similar performance on the test set, with the Decision Tree slightly outperforming the Random Forest in overall accuracy. 


Image.PCA cumulative explained variance. 
The cumulative explained variance increases gradually with additional principal components, reaching about 43% at 80 components. This indicates that variance is spread across many features rather than concentrated in a few dimensions. 


Table 1. Misclassification Analysis:
The model correctly classified 86,665 reviews. The majority of classification errors were false negatives (22,248 cases), which suggests that the model struggled to identify some genuinely helpful reviews. In contrast, false positives were less (7,115 cases), which indicates a tendency to favor predictions of the non-helpful class.


**3. Methods Section**

**3.1 Data Exploration**

The dataset was loaded into Spark and inspected to understand its structure, size, and missing values.
The schema was printed to identify columns types, and summary statistics were computed for numerical and categorical variables. Aggregations were used to explore patterns across product categories, including number of reviews, average rating, and average helpful votes. 

**3.2	Preprocessing**

Data cleaning was performed by removing rows with missing values in key columns such as verified_purchase, product_category, review_body, star_rating, total_votes, and helpful_votes were removed. The rating and vote columns were converted to double precision. A review length feature was generated from the review text, and a binary target variable (helpful_label) was created, where reviews with at least one helpful vote were assigned a value of 1 and all others were assigned a value of 0.

**3.3	Model 1**
As for text preprocessing, it was performed using a Spark ML pipeline that included Tokenization, StopWords removal, and text features were converted into numerical feature vectors using HashingTF with 500 features. TF-IDF weighting was then applied. Principal Component Analysis (PCA) reduced the feature space to 80 principal components. 

Decision Tree with a maximum depth of 5 was trained using the PCA features. 

	The dataset was sampled at 1% of the original size and split into 80% of training data and 20% of testing data. 

**3.4	Model 2**

The same preprocessing and PCA pipeline were used, but the classifier was replaced with a Random Forest model. The Random Forest classifier was configured with 20 trees and trained using the PCA-transformed features.
Both models were evaluated using classification accuracy on the training and testing datasets. PCA explained variance was also recorded to measure how much information was retained after dimensionality reduction. 

**4.	Results Section**

In this section, we present the outputs of the data processing pipeline and models. 

# Class Distribution
- Not Helpful (0): majority class
- Helpful (0): minority class

# PCA Dimenstionality Reduction
- Number of PCA components: 80
- Total explained variance: 43.02%

  
# Model Performance (Accuracy):
Decision Tree: 
-	Training Accuracy: 0.746
-	Test Accuracy: 0.748
-	Training Error: 0.253
-	Test Error: 0.251
  
Random Forest: 
-	Training Accuracy: 0.742
-	Test Accuracy: 0.7429
- Training Error: 0.257
-	Test Error: 0.257

# Confusion Matrices (Decision Tree):
- True Negatives: 75,989
- False Positives: 5,979
- False Negatives: 23,252
- True Positives: 10,808

# Confusion Matrices (Random Forest):
- True Negatives: 77,404
- False Positives: 4,564
- False Negatives: 25,271
- True Positives: 8,789

# Runt-time Performance
- Training time: ~3.66 minutes
- Speedup: 1.04x
- Efficiency: 1.04


**5.	Discussion Section**

The results indicate a moderate classification performance for both models, with Decision Tree slightly outperforming Random Forest. However, these results should be interpreted with caution. A 1% stratified sample was used for computational efficiency, which may limit generalizability to the full dataset. Additionally, the target variable (“helpful_label”) is derived from a simple threshold on helpful votes, which may introduce an inherently learnable pattern.  
While PCA helped reduce dimensionality and improve computational performance, it also reduced interpretability of the text features. In general, although the results suggested reliable model performance, they may partially reflect the structure of the data rather than true predictive complexity. Further work using more balanced data and richer text representations could provide a more robust evaluation. 

**6.	Conclusion**

From this project, we learned how important big data processing is when it comes to working with large-scale datasets. A lot of the work wasn’t just about building models, but about cleaning the data properly, transforming text into usable features, and making sure everything runs efficiently at scale. 
Using distributed computing with Spark also changed how we think about machine learning workflows. For instance, instead of focusing only on accuracy, we had to think also more about how the data is processed and whether the pipeline can handle large volumes of data without slowing down or breaking.
If we had more time, we would have tried better text representations like Word2Vec or transformer-based embeddings since they capture meaning better than TF-IDF. Future work could include more stronger models like gradient boosting and spend time tuning parameters and dimensionality reduction to see if we could get better performance.  


**7.	Statement of Collaboration**

Throughout the entirety of our project Zaineh and I communicated very effectively given that there is about a 10-hour difference between the two of us. For better or for worse we never formally anointed one of us as the designated team leader, project manager, coder or writer but instead did all of the roles together. 
The usual breakdown of how we went about doing our project was I’d propose the idea and if we agreed I’d then do about 75-85% of the coding for it, send it over to Zaineh who would finish the rest of the code and then would do about 90% of the write-ups for the milestone and then send it back to me to recheck. In all honesty I do believe that Zaineh contributed more to the project than I did, Zaineh did a lot of the tedious work such as setting up the github, setting up the configurations for the project, picking out the dataset, rechecking my write-ups and cleaning up any parts of the code I sent in on top of write up’s and more code. Overall, both Zaineh and I contributed heavily to our project and both us dabbled in every role that the project needed.


