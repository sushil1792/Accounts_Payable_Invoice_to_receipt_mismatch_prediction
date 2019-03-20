**INTRODUCTION**

This study investigates the likelihood of a mismatch in receipt and invoice matching processes for payables, called &quot;receipt grief&quot;. An American Fortune 100 company, which is also the world&#39;s largest construction equipment manufacturer, was used as the subject of the study. Our study demonstrates how receipt grief costs can be reduced without manual intervention by developing a supervised learning model that predicts the likelihood of this event. The likelihood prediction is based on invoice attributes such as supplier code, freight transportation mode, location of receiving facility, freight weight and freight charge. If the likelihood of receipt grief is sufficiently large, an invoice can be returned to the receiving facility for review and resolution of the receipt grief before it enters the payables application. This path requires lower overall efforts and less time while reducing charges associated with manual review by freight analysts. We explore several categorical variable imputation methods like Multivariate Imputation by Chained Equations (MICE), nearest neighbor and mode imputation to reduce information loss. We implemented and compared the performance of four supervised learning algorithms namely, logistic regression, decision tree, random forest, and XG Boost using scikit-learn package in python.

**DATA**

Invoice data was provided by the manufacturing company over a two-year time-frame. Operational parameters such as grief code status, invoice codes, entry dates, bill line types were provided. The data consisted of 7 million rows of mainly categorical variables (Only 2 out of 20 columns were numerical). Here, the &quot;GRIEF\_CODE&quot; is our dependent (y) variable and the remaining variables are the independent (x) variables.

Given below in Table 1 is a comprehensive explanation of the individual variables:

**1**

**2**

The data analysis and modeling procedures used in the study is illustrated in Figure 1. The key steps involved for data analysis and modeling include: Data preparation involving Data Cleaning, Feature Engineering and Data Pre-processing, Data Modelling and Model Evaluation.

**EXPLORATORY DATA ANALYSIS**

We investigated the characteristics of variables contained within the data set, including their data types, range of valid values, their distributions, and missing data values. Results of data exploration were used to develop strategies for modeling. The following are some of the important features we explored:

1. The time series plots in figure 2, grief (grief varying with ship date and entry date) showed no significant seasonality, cycles or trends. One of the significant models that we considered applying was the time series model. However, because there were no interesting patterns observed, we did not analyze the data using this model.

**3**

2. In Figure 3, we analyzed the distribution of facilities based on facility types for all the records in which grief has been recorded. It is observed that Facility Types &quot;S&quot; and &quot;F&quot; have the highest number of facilities generating receipts with grief. Another observation is that across all the facility types, the largest portion of the transportation of materials/supplies has taken place through transport mode &quot;T&quot; followed by &quot;A&quot;. We have also found the distribution of receipt grief across the mode of receipt, namely electronic and paper. It is observed that on average, for electronic receipt the number of cases where grief has been observed is much higher compared to paper (potentially due to manual issues in the paper receipts). Moreover, most of the receipts belong to the Facility Types &quot;S&quot; and &quot;F&quot;.

**4**

3. We observed that the grief is majorly observed for high freight charge for all types of freight bill. This provided us with an indication that the freight charge may potentially be an indicator of grief.

**5**

The provided data contained missing values and values that were significantly difficult to interpret. Thus, before beginning our analysis, we had to perform several rounds of data cleaning to ensure better performance of our predictive models.

1. There were several different categories grief namely B- BOL Grief, C- Communication Grief, M- MM Facility, P- MM Carrier and R- Receipt in the data. However, we were only interested in finding the possibility of the occurrence of &quot;receipt grief&quot;. Hence, we removed the rows corresponding to the other types of grief.

2. We had data spanning a period of 2 years (8 quarters) from 2011 through 2012 that had two date variables viz. &quot;SHIP\_DATE&quot; &amp; &quot;ENTRY\_DATE&quot;. We tried to find receipt grief patterns (seasonality or trends) using Tableau. However as mentioned in the Exploratory Data Analysis section, these columns were not useful in any way for understanding the occurrence of receipt grief as there was no noticeable seasonality or trend in the data. Hence, we removed both columns.

3. Receipt grief is only possible in case of received freights/shipments and not for outward shipping. Hence the data was filtered using this criterion on the column &quot;FRT\_IN\_OUT\_IND&quot; (where 1 indicates inbound shipments).

4. The column &quot;EDI\_IND&quot; corresponding to the receipt being electronic or paper based had an unconventional coding standard (with blanks indicating paper). Thus, we replaced all blanks with &quot;0&quot; to maintain readability across the whole dataset.

5. The dataset also consisted of predictors (&quot;DOCK\_NO&quot;, and &quot;FRT\_CLRK\_CD&quot;) that were perfectly correlated with the dependent variable (receipt grief). A perfect correlation suggested that these predictors were codes that were assigned after the occurrence of grief, therefore, these features were removed from the model.

6. The missing values in the remaining features were imputed. Since, the most important aspect for our study was the imputation of the categorical missing values, we cover it in detail in the section that follows:

Handling Missing Values

&quot;Values in a data set are missing completely at random (MCAR) if the events that lead to any particular data-item being missing are independent both of observable variables and of unobservable parameters of interest and occur entirely at random&quot;[[4]](https://en.wikipedia.org/wiki/Missing_data). We assume that the missing data is a result of system-related issues. Consequently, we assume that the data is missing completely at random (MCAR). Since missing data introduce some ambiguity to the overall analysis approach, it is important to identify the correct approach for handling missing data. We considered several studies that helped us understand the correct approach. Overall, our data has two fundamental characteristics:

1. Our variables are mainly categorical,

2. Our data is missing completely at random (MCAR) and,

3. Our dataset has 3.5 million rows

We considered several methods for variable imputation including Multiple Imputations by Chained Equations (MICE), Mode imputation and K-Nearest Neighbors (KNN) imputation.

Multiple Imputations by Chained Equations (MICE): MICE has emerged as one of the primary methods for dealing with missing data. MICE, however, operates under the assumption that given the variables used in the imputation procedure, the missing data, is Missing at Random (MAR). This means that it assumes that the probability that a value is missing depends only on observed values and not on unobserved values.

K-Nearest Neighbors (KNN): KNN is another interesting method that is used for dealing with categorical missing data. KNN assumes that a missing data point can be approximated by the points closest to it. The Nearest Neighbor algorithm can also be more efficiently extended for categorical data by weighting of attributes[[1]](https://arxiv.org/pdf/1710.01011.pdf).

While both MICE and KNN are robust techniques used in practice, for our purposes, both these methods did not perform well. Due to the large number of rows categorical data involved and the number of factor levels in each variable, it became computationally intensive to use KNN and MICE imputation methods. As the study by Peter Schmitt, Jonas Mandel and Mickael Guedj[
# [3]
](https://www.omicsonline.org/open-access/a-comparison-of-six-methods-for-missing-data-imputation-2155-6180-1000224.php?aid=54590) confirms, the performance of the MICE algorithm (which is complex in nature) appears to be directly related to the size of the dataset. The performance of the algorithm decreases as the size of the dataset increases. It is also important to note that MICE assume that the data is Missing at Random (MAR), which is not the same assumption made in our study (we assume the data is Missing Completely at Random (MCAR)). Implementing MICE when data is not missing at random could possibly result in biased estimates.[[2]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074241/)

Using Measures of Central Tendency: Because computational intensity was the major issue in our study, we used mode imputation for categorical data. Here, we used the &quot;ImputeMissings&quot; package in R to replace the missing values for each variable and then replaced the missing values with the mode values for those variables respectively. While mode imputation has its trade-offs (it often results in relatively biased estimates), it was the simplest and the least computationally intensive method for our study. When the median/mode in the ImputeMissings method is used, character vectors and factors are imputed with the mode and numeric and integer vectors are imputed with the median.

**FEATURE ENGINEERING**

The following processes were performed as a part of the feature engineering section:

1. Category Binning: We had several variables that had numerous factor levels (\&gt;1000). These levels had to be combined (binned) to reduce computational complexity while modeling.  The variables that required binning were &quot;SUPP\_DLR\_FAC\_CD&quot;, &quot;ORIG\_CITY\_STATE&quot;, &quot;DEST\_CITY\_STATE&quot;, &quot;CORP\_CARR\_CD&quot;, &quot;FRT\_BILL\_TYP&quot;, &quot;FAC\_CD&quot;, &quot;DOCK\_NO&quot;, &quot;FRT\_CLRK\_CD&quot;.

Binning based on frequency distribution: There are broadly two important methods that can be used for binning: 1) Using Business Logic, 2) Using frequency distribution. Using Business logic, we can combine factor levels based on &quot;domain knowledge&quot;, however, for our study, it was more appropriate to use the second method, i.e. frequency distributions to combine factor levels.

For each variable, we generated the frequency distribution for each factor level with respect to the receipt grief. The top 10 (most frequent) levels were left as they are, and the others were combined into a new level called &quot;others&quot;.

2. We applied min-max normalization on the numeric features viz. &quot;FRT\_TRN\_WT\_LB&quot; and &quot;FRT\_CHRG&quot;. This was done to ensure uniformity and readability across the dataset and to eliminate negative (discount) values in &quot;FRT\_CHRG&quot;.

3.We created more than 90 dummy variables for the new factors that were formed in the dataset.

**MODELING: SELECTION AND COMPARISON**

For our analysis, we first split the dataset into training and testing datasets (80:20 ratio) and chose four types of supervised modeling techniques to predict the receipt grief. We used the test dataset to validate the performance of each model and evaluated and compared these models based on the accuracy, AUC scores with specific importance placed in the specificity of the model. Additionally, we also downsampled the non grief observations to reduce the skewness in the classes achieving equal class distribution for each of the prediction class.  Figure 5 illustrates the skewness, which is resolved with downsampling.

**6**

Logistic Regression: It is the simplest and the most interpretable model that is used to predict an outcome variable that is categorical from predictor variables that are continuous and/or categorical. We wanted to start with the simplest model assess its performance and then move on to the more complex modeling techniques. Generally, it is used because having a categorical outcome variable violates the assumption of linearity in normal regression. The only &quot;real&quot; limitation for logistic regression is that the outcome variable must be discrete and logistic regression deals with this problem by using a logarithmic transformation on the outcome variable which allow us to model a nonlinear association in a linear way.

Decision Trees: We used decision trees as it was suggested by the company as a model that is used in the standard course of business. Decision tree learninguses a [decision tree](https://en.wikipedia.org/wiki/Decision_tree) (as a [predictive model](https://en.wikipedia.org/wiki/Predictive_modelling)) to go from observations about an item (represented in the branches) to conclusions about the item&#39;s target value (represented in the leaves). It is one of the important predictive modeling approaches used in [statistics](https://en.wikipedia.org/wiki/Statistics), [data mining](https://en.wikipedia.org/wiki/Data_mining) and [machine learning](https://en.wikipedia.org/wiki/Machine_learning). Tree models where the target variable can take a discrete set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels.

Random Forests: We decided to go one step ahead from decision trees and use Random Forest. The previous model (i.e. the decision tree) is essentially a building block for the random forest model. Random Forests grow many classification trees. To classify a new object from an input vector, put the input vector down each of the trees in the forest. Each tree gives a classification, and we say the tree &quot;votes&quot; for that class. The forest chooses the classification having the most votes (over all the trees in the forest).

XGBoost: Along with the random forest, we decided to also use the XGBoost model, which is again an ensemble of decision trees. Gradient boosting is a machine learning technique for classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion like other boosting methods do, and it generalizes them by allowing optimization of an arbitrary differentiable loss function. XGBoost is one of the implementations of Gradient Boosting concept, but what makes XGBoost unique is that it uses a more regularized model formalization to control over-fitting, which gives it better performance. We chose this model to further improve our results.

Results

Overall, the random forest performed well on the test set, with an overall accuracy of 0.87 and an AUC score of 0.92. Here, the specificity is also of vital importance because it shows how many times the model correctly classified the actual occurrence of the receipt grief event. Random forest showed the highest specificity of 0.96 (along with decision trees). Table 2 compares the model evaluation parameters for all the models used and Figure 5 compares the AUC scores of the models.

**7**

Description of random forest results: The random forest model also showed the following feature importance graph (Figure 6). Here we see that &quot;FRT\_CHRG&quot; shows the maximum importance of 23.3, this means that 23.3% of the variability in the dependent variable (&quot;GRIEF\_CODE&quot;) can be explained by &quot;FRT\_CHRG&quot; alone. The other important features from this model are: &quot;FRT\_TRN\_WT\_LB&quot;, which explains 11.1% of the variability in the dependent variable, &quot;SUPP\_DLR\_FAC\_TYP&quot; (specifically type &quot;s&quot;, which has been encoded due to confidentiality concerns).

**8                      **

Once we have identified the best performing model and the important features that will help predict receipt grief, it is important for us to interpret these results further from a business perspective and provide the company with a solution that can help reduce the costs associated with receipt grief. This portion will be covered in the &quot;Recommendations and Insights&quot; section that follows.

**RECOMMENDATIONS &amp; INSIGHTS**

We recommended the following to the company to improve the performance of the model and the internal processes to reduce costs associated with receipt grief:

1. While we know the features that caused receipt grief, the company could delve into further detail and granularize the data. If we know that the freight charge is causing maximum receipt grief, the company could identify the underlying issue. From our analysis, we believe that the freight charge could be responsible for a significant number of receipt grief cases because there could have potentially been differences in the exchange rates while the supplier sent the invoice and the buyer processed the receipt (the exchange rates used in calculations of values could be different). Similarly, a difference in the freight weight could imply that the supplier&#39;s invoice contained a different material weight, but the material was lost/damaged after it was shipped. The supplier facility type (specifically type &quot;s&quot;) was also an important indicator of receipt grief. This could be due to several issues at the suppliers&#39; end that results in a mismatch of the contents illustrated in the invoice and the materials received at the receiving facility. A thorough investigation could be conducted to understand the underlying issues at the receiving facility and at the suppliers&#39; end. This would help combat the problem of receipt grief.

2. The current process involves a complex matching payables application for invoice and receipts after the receipt is created. There is also a time lag between when the invoice is received at the receiving facility and when the receipt is generated. Our recommendation would be to deploy our predictive model to predict the receipt grief as soon as the invoice is received. If the probability is high (as per the standard threshold determined by the company), the company could send the invoice back quickly to the receiving facility, reducing the 20-day time-frame defined in the current matching process. If the receipt grief is resolved at the receiving facility immediately, without moving to the account payables team, the issues could be rectified easily and the time and cost for resolution can be significantly reduced.

**9**

Since the best predictive model is 87% accurate, the total costs for receipt grief can be reduced to 0.13x $450K (current monthly receipt grief costs for line-by-line resolution by the freight analyst), which is $58K, a significant reduction in costs.
