**Milestone 4: Final Submission**



**3. Fitting Analysis:**

• For our second model we receives a training accuracy of ___ and a test accuracy of ___. The two accuracy are in close proximity of each other overfitting doesn't seem to be a huge issue. The model is doing a decent job of learning pattern about the data.

• Potential future improvements: to improve our model we could increase our PCA from 50 to 100 and hashing up to 2500-5000. By increasing the hashing we allow for wider variety of word patterns which the PCA can then compress into a valuable set of words, after the idf is applied. We could also potentially switch to using boosted trees to pick up complex patterns that weren't identified in the previous model.

• Dimensionality Reduction: Dimensionality reduction helped us gather insights on the review text but only on text that was deemed important. If we tried to test on every word in the review text the code would've probably crashed. Using a PCA-based model allowed us to pick the best words that had the most accuracy, which made modeling a lot easier.













# Data URL & Code
- **Dataset:** https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset
- **Notebook:** https://github.com/Zeina08/Amazon-US-Customer-Reviews-Dataset/blob/main/milestones/ReviewDataset.ipynb

# Environment:
- **Platform:** SDSC Expanse (ACCESS CI allocation portal)
- **Framework:** PySpark
