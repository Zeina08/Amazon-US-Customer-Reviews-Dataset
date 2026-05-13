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
There are 58075027 rows in the dataset.

**b.** Describe all columns in your dataset: their scales and data distributions. Describe categorical and continuous variables. Describe your target column.

Categorical variables (string):
-	Marketplace: US
-	Review_id: identifier for each review (unique)
-	Product_id: identifier for products
-	Product_category: books, video games, etc
-	Vine: previews whether a review is part of Amazon vine program (binary as in y/n)
-	Verified_purchase: if purchases are verified 
-	Review_headline: short text
-	Review_body: full review text
Numerical:
-	Customer_id: numerical id
-	Product_parent: related to products(grouping)
-	Star_rating: ratings by users from 1-5
-	Helpful_votes: number of users who found reviews to be helpful
-	Total_votes: total votes on reviews
Time-based variable: 
-	Review_date: when reviews were posted
Main target column:
-	Star_rating: represents user sentiment (1-5)
Through the describe output for the star_rating: the distribution is highly skewed towards 4-5 stars (most values clustering near the high end).

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

















