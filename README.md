**Dataset:https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset**

**2.** **SDSC Expanse Environment Setup:**
a. Expanse login/access: through web portal (accessed resources through ACCESS CI -> allocation portal)
Environement: ran work in jupyter: python with pyspark

**b.** **SparkSession configuration:**
TOTAL_MEMORY = 64
TOTAL_CORES = 8
DRIVER_MEMORY = 2
Justification:
-	Driver memory is kept smaller as it focuses on coordination and not computation
-	Seven executors more efficient for parallel parallelism across the present cores
  
**c. Formula:**
Total cores -1
8 – 1 = 7 executors
Executor memory = total memory – driver memory / num executors
62-2 / 7 =  (around 8GB)

**3. Data Exploration using Spark**:

**a.** How many observations does your dataset have?
There are 58075027 rows in the datase and 15 columns (features).

**b.** Describe all columns in your dataset: their scales and data distributions. Describe categorical and continuous variables. Describe your target column.

Categorical variables (string):
-	Marketplace: US
-	Product_category: books, video games, etc
-	Vine: previews whether a review is part of Amazon vine program (binary as in y/n)
-	Verified_purchase: if purchases are verified 
-	Review_headline: short text
-	Review_body: full review text
Numerical (continuous):
-	Star_rating: ratings by users from 1-5
-	Helpful_votes: number of users who found reviews to be helpful
-	Total_votes: total votes on reviews
Identifier variables:
-	Review_id: identifier for each review (unique)
-	Product_id: identifier for products
- Product_parent: related to products(grouping)
- Customer_id: numerical id
  
Time-based variable: 
-	Review_date: when reviews were posted
Main target column:
-	Star_rating: represents user sentiment (1-5)
Through the describe output for the star_rating: the distribution is highly skewed towards 4-5 stars (most values clustering near the high end).

In terms of columns, the dataset contains customer information, product attributes, review text, ratings, voting activity, and timestamps used to analyze factors associated with rteview helpfulness. The star_rating ranges from 1-5 (mean: 4.17) showing a distribution skewed toward higher ratings. helpful_votes and total_votes are right skewed count variables, with most reviews receving few votes and a small number receiving very high values. Product categories also vary greatly in scale, with review counts ranging in the millions. The data shows signs of imbalance with star_rating heavily concentrated in 4-5 star reviews, indicating positive reviews are much more common than lower-rated ones. 

**c.** Do you have missing and duplicate values in your dataset?
Yes, for missing: total missing rows: 8953 
As for duplicate rows we did perform in the first edited jupyter (took too much space on the second attempt). duplicate rows: 58075027 - 58038800 = 36,227

**4. Data Plots:**

**d.** Clearly explain each plot and what insights it provides
We conducted two plots:
1.	The bar chart shows the top 10 product categories based on the number of reviews. The wireless and PC products are the top two categories, indicating that they received the highest level of user engagement. This also suggests that technology products are more frequently reviewed compared to other categories. On the other hand, video games have the lowest number of reviews among the top categories. Overall, the results show an uneven distribution of reviews across product_types.
2.	Scatterplot:  suggests that star rating by itself doesn’t strongly predict how many helpful votes a review does receive. Most reviews cluster at relatively low helpful vote counts, which indicates that the majority don’t gain much traction. However, there are a few clear outliers: 4-star review with very high helpful votes and 1 star review with a large count as well. This could imply that while both positive and negative reviews can attract attention, usefulness is likely driven more by other factors we are yet to explore. 

**5. Preprocessing Plan:**
For our project, we will handle any null values by dropping them from the dataset to ensure that we don’t create imbalances. Since different variables may have different amounts of missing data, we will downsample to match the lowest independent variable count in order to maintain an unbiased dataset.
For transformations, we will use a OneHotEncoder to convert product categories into binary variables so that any machine learning methods we use will be effective. We will also create beneficial variables such as review length. This can help us because reviews that are only one or two letters are likely not legitimate or may be considered “troll” reviews, and they can be excluded from the machine learning process. This will allow us to achieve better accuracy when determining whether a review is actually helpful or not.
For the actual Spark operations, we will use dropna(),filter(), withColumn(), OneHotEncoder, a possible scaler, and other basic Python and SQL operations.

**Fitting Analysis:**
Training accuracy: ~ 0.929 vs Test accuracy: ~ 0.93 (small gap)
Training and test accuracy results are almost identical, showing good generalization. There is no sign of overfitting or underfitting (low accuracy). Which means the model is learning patterns without memorizing the data. 

**Another model testing with different hyperparameters:**
Tree-based models (Decision Tree and Random Forest) were tested using different hyperparameters. Despite Random Forest being more complex, it did not show any improvement over the Decision Tree. This suggests that the Decision Tree already captures most of the predictive patterns in the data. 

**Which model performs best and why?**
Both models perform equally well. The Random Forest doesn't significantly outperform the Decision Tree. Both models generalize well and achieve the same accuracy. Overall, the Decision Tree is the preferred model because it provides the same performance with lower complexity and is faster when it comes to training data.

**What are the next models you are thinking of for Milestone 4 and why?**
Fro Milestone 4, the next step is to incorporate natural language processing (NLP) techniques to better analyze review text. Models such as TF-IDF with other advanced approaches will be explored to capture sentiment and contextual information/patterns within the reviews. Using spark will be efficient for large-scale text processing models. 

**Conclusion Section**
**What is the conclusion of your 1st model?**

Based on our first model, we found our Decision Tree to have around a .93 accuracy for both training and testing data. Given an accuracy of .93, our model does a fairly decent job in predicting whether a customer review is regarded as helpful

**What can be done to possibly improve it?**
To improve our model, a technique that could be is a possible tokenization of the customer review text. By tokenizing we allow ourselves to see a new perspective of the data that was once unavailable. From the tokenization, if we keep words such as "good", "bad" "helpful" "convenient", "great" we can still get all of the benefits of tokenization without making the decision tree unnecessarily large and overfitting. 

**How did distributed computing help with this task?**
Distributed computing was very helpful in completing our task because it allowed us to split data filtering, cleaning, and computation onto multiple nodes. This parallel processing allows for efficient processing that would not be done on a singular device and given a dataset of 50GB, would most likely cause the device to crash.

















