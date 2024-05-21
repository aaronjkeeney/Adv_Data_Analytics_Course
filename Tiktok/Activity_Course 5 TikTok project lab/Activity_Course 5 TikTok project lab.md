# **TikTok Project**
**Course 5 - Regression Analysis: Simplify complex data relationships**

You are a data professional at TikTok. The data team is working towards building a machine learning model that can be used to determine whether a video contains a claim or whether it offers an opinion. With a successful prediction model, TikTok can reduce the backlog of user reports and prioritize them more efficiently.

The team is getting closer to completing the project, having completed an initial plan of action, initial Python coding work, EDA, and hypothesis testing.

The TikTok team has reviewed the results of the hypothesis testing. TikTok’s Operations Lead, Maika Abadi, is interested in how different variables are associated with whether a user is verified. Earlier, the data team observed that if a user is verified, they are much more likely to post opinions. Now, the data team has decided to explore how to predict verified status to help them understand how video characteristics relate to verified users. Therefore, you have been asked to conduct a logistic regression using verified status as the outcome variable. The results may be used to inform the final model related to predicting whether a video is a claim vs an opinion.

A notebook was structured and prepared to help you in this project. Please complete the following questions.

# **Course 5 End-of-course project: Regression modeling**


In this activity, you will build a logistic regression model in Python. As you have learned, logistic regression helps you estimate the probability of an outcome. For data science professionals, this is a useful skill because it allows you to consider more than one variable against the variable you're measuring against. This opens the door for much more thorough and flexible analysis to be completed.

<br/>

**The purpose** of this project is to demostrate knowledge of EDA and regression models.

**The goal** is to build a logistic regression model and evaluate the model.
<br/>
*This activity has three parts:*

**Part 1:** EDA & Checking Model Assumptions
* What are some purposes of EDA before constructing a logistic regression model?

**Part 2:** Model Building and Evaluation
* What resources do you find yourself using as you complete this stage?

**Part 3:** Interpreting Model Results

* What key insights emerged from your model(s)?

* What business recommendations do you propose based on the models built?

Follow the instructions and answer the question below to complete the activity. Then, you will complete an executive summary using the questions listed on the PACE Strategy Document.

Be sure to complete this activity before moving on. The next course item will provide you with a completed exemplar to compare to your own work.


# **Build a regression model**

<img src="images/Pace.png" width="100" height="100" align=left>

# **PACE stages**

Throughout these project notebooks, you'll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.

<img src="images/Plan.png" width="100" height="100" align=left>


## **PACE: Plan**
Consider the questions in your PACE Strategy Document to reflect on the Plan stage.

### **Task 1. Imports and loading**
Import the data and packages that you've learned are needed for building regression models.


```python
# Import packages for data manipulation
import numpy as np
import pandas as pd

# Import packages for data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Import packages for data preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Import packages for data modeling
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics

```

Load the TikTok dataset.

**Note:** As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.


```python
# Load dataset into dataframe
data = pd.read_csv("tiktok_dataset.csv")
```

<img src="images/Analyze.png" width="100" height="100" align=left>

## **PACE: Analyze**

Consider the questions in your PACE Strategy Document to reflect on the Analyze stage.

In this stage, consider the following question where applicable to complete your code response:

* What are some purposes of EDA before constructing a logistic regression model?


We want to ensure that the data are complete and in the correct format for us to use. We also want to re-familiarize ourselves with the data in order to have a clear understanding of what it included. This will help us decide which variables to include as part of our logistic regression model.

### **Task 2a. Explore data with EDA**

Analyze the data and check for and handle missing values and duplicates.

Inspect the first five rows of the dataframe.


```python
# Display first few rows
data.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>claim_status</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_transcription_text</th>
      <th>verified_status</th>
      <th>author_ban_status</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>claim</td>
      <td>7017666017</td>
      <td>59</td>
      <td>someone shared with me that drone deliveries a...</td>
      <td>not verified</td>
      <td>under review</td>
      <td>343296.0</td>
      <td>19425.0</td>
      <td>241.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>claim</td>
      <td>4014381136</td>
      <td>32</td>
      <td>someone shared with me that there are more mic...</td>
      <td>not verified</td>
      <td>active</td>
      <td>140877.0</td>
      <td>77355.0</td>
      <td>19034.0</td>
      <td>1161.0</td>
      <td>684.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>claim</td>
      <td>9859838091</td>
      <td>31</td>
      <td>someone shared with me that american industria...</td>
      <td>not verified</td>
      <td>active</td>
      <td>902185.0</td>
      <td>97690.0</td>
      <td>2858.0</td>
      <td>833.0</td>
      <td>329.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>claim</td>
      <td>1866847991</td>
      <td>25</td>
      <td>someone shared with me that the metro of st. p...</td>
      <td>not verified</td>
      <td>active</td>
      <td>437506.0</td>
      <td>239954.0</td>
      <td>34812.0</td>
      <td>1234.0</td>
      <td>584.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>claim</td>
      <td>7105231098</td>
      <td>19</td>
      <td>someone shared with me that the number of busi...</td>
      <td>not verified</td>
      <td>active</td>
      <td>56167.0</td>
      <td>34987.0</td>
      <td>4110.0</td>
      <td>547.0</td>
      <td>152.0</td>
    </tr>
  </tbody>
</table>
</div>



Get the number of rows and columns in the dataset.


```python
# Get number of rows and columns
print('(rows, columns)', data.shape)

```

    (rows, columns) (19382, 12)


Get the data types of the columns.


```python
# Get data types of columns
data.dtypes

```




    #                             int64
    claim_status                 object
    video_id                      int64
    video_duration_sec            int64
    video_transcription_text     object
    verified_status              object
    author_ban_status            object
    video_view_count            float64
    video_like_count            float64
    video_share_count           float64
    video_download_count        float64
    video_comment_count         float64
    dtype: object



Get basic information about the dataset.


```python
# Get basic information
data.info()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 19382 entries, 0 to 19381
    Data columns (total 12 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   #                         19382 non-null  int64  
     1   claim_status              19084 non-null  object 
     2   video_id                  19382 non-null  int64  
     3   video_duration_sec        19382 non-null  int64  
     4   video_transcription_text  19084 non-null  object 
     5   verified_status           19382 non-null  object 
     6   author_ban_status         19382 non-null  object 
     7   video_view_count          19084 non-null  float64
     8   video_like_count          19084 non-null  float64
     9   video_share_count         19084 non-null  float64
     10  video_download_count      19084 non-null  float64
     11  video_comment_count       19084 non-null  float64
    dtypes: float64(5), int64(3), object(4)
    memory usage: 1.8+ MB


Generate basic descriptive statistics about the dataset.


```python
# Generate basic descriptive stats
data.describe()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>19382.000000</td>
      <td>1.938200e+04</td>
      <td>19382.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9691.500000</td>
      <td>5.627454e+09</td>
      <td>32.421732</td>
      <td>254708.558688</td>
      <td>84304.636030</td>
      <td>16735.248323</td>
      <td>1049.429627</td>
      <td>349.312146</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5595.245794</td>
      <td>2.536440e+09</td>
      <td>16.229967</td>
      <td>322893.280814</td>
      <td>133420.546814</td>
      <td>32036.174350</td>
      <td>2004.299894</td>
      <td>799.638865</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.234959e+09</td>
      <td>5.000000</td>
      <td>20.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4846.250000</td>
      <td>3.430417e+09</td>
      <td>18.000000</td>
      <td>4942.500000</td>
      <td>810.750000</td>
      <td>115.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9691.500000</td>
      <td>5.618664e+09</td>
      <td>32.000000</td>
      <td>9954.500000</td>
      <td>3403.500000</td>
      <td>717.000000</td>
      <td>46.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>14536.750000</td>
      <td>7.843960e+09</td>
      <td>47.000000</td>
      <td>504327.000000</td>
      <td>125020.000000</td>
      <td>18222.000000</td>
      <td>1156.250000</td>
      <td>292.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>19382.000000</td>
      <td>9.999873e+09</td>
      <td>60.000000</td>
      <td>999817.000000</td>
      <td>657830.000000</td>
      <td>256130.000000</td>
      <td>14994.000000</td>
      <td>9599.000000</td>
    </tr>
  </tbody>
</table>
</div>



Check for and handle missing values.


```python
# Check for missing values
data.isna().sum()
```




    #                             0
    claim_status                298
    video_id                      0
    video_duration_sec            0
    video_transcription_text    298
    verified_status               0
    author_ban_status             0
    video_view_count            298
    video_like_count            298
    video_share_count           298
    video_download_count        298
    video_comment_count         298
    dtype: int64




```python
# Drop rows with missing values
data = data.dropna(axis = 0).reset_index(drop = True)
```


```python
# Display first few rows after handling missing values
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>claim_status</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_transcription_text</th>
      <th>verified_status</th>
      <th>author_ban_status</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>claim</td>
      <td>7017666017</td>
      <td>59</td>
      <td>someone shared with me that drone deliveries a...</td>
      <td>not verified</td>
      <td>under review</td>
      <td>343296.0</td>
      <td>19425.0</td>
      <td>241.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>claim</td>
      <td>4014381136</td>
      <td>32</td>
      <td>someone shared with me that there are more mic...</td>
      <td>not verified</td>
      <td>active</td>
      <td>140877.0</td>
      <td>77355.0</td>
      <td>19034.0</td>
      <td>1161.0</td>
      <td>684.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>claim</td>
      <td>9859838091</td>
      <td>31</td>
      <td>someone shared with me that american industria...</td>
      <td>not verified</td>
      <td>active</td>
      <td>902185.0</td>
      <td>97690.0</td>
      <td>2858.0</td>
      <td>833.0</td>
      <td>329.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>claim</td>
      <td>1866847991</td>
      <td>25</td>
      <td>someone shared with me that the metro of st. p...</td>
      <td>not verified</td>
      <td>active</td>
      <td>437506.0</td>
      <td>239954.0</td>
      <td>34812.0</td>
      <td>1234.0</td>
      <td>584.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>claim</td>
      <td>7105231098</td>
      <td>19</td>
      <td>someone shared with me that the number of busi...</td>
      <td>not verified</td>
      <td>active</td>
      <td>56167.0</td>
      <td>34987.0</td>
      <td>4110.0</td>
      <td>547.0</td>
      <td>152.0</td>
    </tr>
  </tbody>
</table>
</div>



Check for and handle duplicates.


```python
# Check for duplicates
data.duplicated().sum()
```




    0



Check for and handle outliers.


```python
# Create a boxplot to visualize distribution of `video_duration_sec`
sns.boxplot(data = data, x = 'video_duration_sec')


```




    <Axes: xlabel='video_duration_sec'>




    
![png](output_31_1.png)
    



```python
# Create a boxplot to visualize distribution of `video_view_count`
sns.boxplot(data = data, x = 'video_view_count')


```




    <Axes: xlabel='video_view_count'>




    
![png](output_32_1.png)
    



```python
# Create a boxplot to visualize distribution of `video_like_count`
sns.boxplot(data = data, x = 'video_like_count')


```




    <Axes: xlabel='video_like_count'>




    
![png](output_33_1.png)
    



```python
# Create a boxplot to visualize distribution of `video_comment_count`
sns.boxplot(data = data, x = 'video_comment_count')


```




    <Axes: xlabel='video_comment_count'>




    
![png](output_34_1.png)
    



```python
# Check for and handle outliers for video_like_count

percentile25 = data['video_like_count'].quantile(0.25)
percentile75 = data['video_like_count'].quantile(0.75)

iqr = percentile75 - percentile25

upper_threshold = 1.5 * iqr + percentile75

print(f"Outlier cutoff: {upper_threshold}")
print(f"Maximum like count: {data['video_like_count'].max()}")

#based on this quick check, there are no outliers to be addressed.
#Furthermore, the nature of viral videos would suggest that large datapoints are important and should be included.

```

    Outlier cutoff: 311333.875
    Maximum like count: 657830.0


Check class balance.


```python
# Check class balance for verified status.
data['verified_status'].value_counts(normalize = True)
```




    verified_status
    not verified    0.93712
    verified        0.06288
    Name: proportion, dtype: float64



Approximately 94% of the dataset represents videos posted by unverified accounts and 6% represents videos posted by verified accounts. So the outcome variable is not very balanced.

Use resampling to create class balance in the outcome variable, if needed.


```python
# Use resampling to create class balance in the outcome variable, if needed

# Identify data points from majority and minority classes
majority_class = data[data['verified_status'] == 'not verified']
minority_class = data[data['verified_status'] == 'verified']

# Upsample the minority class (which is "verified")
upsampled_minority_class = resample(minority_class,replace = True, n_samples=len(majority_class), random_state=0)

# Combine majority class with upsampled minority class
data_upsampled = pd.concat([majority_class, upsampled_minority_class]).reset_index(drop=True)

# Display new class counts
data_upsampled['verified_status'].value_counts(normalize = True)
```




    verified_status
    not verified    0.5
    verified        0.5
    Name: proportion, dtype: float64



Get the average `video_transcription_text` length for videos posted by verified accounts and the average `video_transcription_text` length for videos posted by unverified accounts.




```python
# Get the average `video_transcription_text` length for claims and the average `video_transcription_text` length for opinions
data_upsampled[['video_transcription_text', 'verified_status']].groupby('verified_status')[['video_transcription_text']].agg(
    func = lambda array: np.mean([len(text) for text in array]))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>video_transcription_text</th>
    </tr>
    <tr>
      <th>verified_status</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>not verified</th>
      <td>89.401141</td>
    </tr>
    <tr>
      <th>verified</th>
      <td>84.569559</td>
    </tr>
  </tbody>
</table>
</div>



Extract the length of each `video_transcription_text` and add this as a column to the dataframe, so that it can be used as a potential feature in the model.


```python
# Extract the length of each `video_transcription_text` and add this as a column to the dataframe
data_upsampled['text_length'] = data_upsampled['video_transcription_text'].apply(func = lambda text: len(text))

```


```python
# Display first few rows of dataframe after adding new column
data_upsampled.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>claim_status</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_transcription_text</th>
      <th>verified_status</th>
      <th>author_ban_status</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
      <th>text_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>claim</td>
      <td>7017666017</td>
      <td>59</td>
      <td>someone shared with me that drone deliveries a...</td>
      <td>not verified</td>
      <td>under review</td>
      <td>343296.0</td>
      <td>19425.0</td>
      <td>241.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>97</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>claim</td>
      <td>4014381136</td>
      <td>32</td>
      <td>someone shared with me that there are more mic...</td>
      <td>not verified</td>
      <td>active</td>
      <td>140877.0</td>
      <td>77355.0</td>
      <td>19034.0</td>
      <td>1161.0</td>
      <td>684.0</td>
      <td>107</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>claim</td>
      <td>9859838091</td>
      <td>31</td>
      <td>someone shared with me that american industria...</td>
      <td>not verified</td>
      <td>active</td>
      <td>902185.0</td>
      <td>97690.0</td>
      <td>2858.0</td>
      <td>833.0</td>
      <td>329.0</td>
      <td>137</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>claim</td>
      <td>1866847991</td>
      <td>25</td>
      <td>someone shared with me that the metro of st. p...</td>
      <td>not verified</td>
      <td>active</td>
      <td>437506.0</td>
      <td>239954.0</td>
      <td>34812.0</td>
      <td>1234.0</td>
      <td>584.0</td>
      <td>131</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>claim</td>
      <td>7105231098</td>
      <td>19</td>
      <td>someone shared with me that the number of busi...</td>
      <td>not verified</td>
      <td>active</td>
      <td>56167.0</td>
      <td>34987.0</td>
      <td>4110.0</td>
      <td>547.0</td>
      <td>152.0</td>
      <td>128</td>
    </tr>
  </tbody>
</table>
</div>



Visualize the distribution of `video_transcription_text` length for videos posted by verified accounts and videos posted by unverified accounts.


```python
# Visualize the distribution of `video_transcription_text` length for videos posted by verified accounts and videos posted by unverified accounts
# Create two histograms in one plot
sns.histplot(data = data_upsampled, x = 'text_length', multiple = 'dodge', hue = 'verified_status')
plt.title('Histogram of Text Length for Verified vs. Non-verified Users')
plt.xlabel('Text Length (total characters)')
plt.ylabel('Number of Videos')
plt.show()

```


    
![png](output_47_0.png)
    


### **Task 2b. Examine correlations**

Next, code a correlation matrix to help determine most correlated variables.


```python
# Code a correlation matrix to help determine most correlated variables
data_upsampled.corr(numeric_only = True)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
      <th>text_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>#</th>
      <td>1.000000</td>
      <td>-0.000853</td>
      <td>-0.011729</td>
      <td>-0.697007</td>
      <td>-0.581483</td>
      <td>-0.504015</td>
      <td>-0.487096</td>
      <td>-0.413799</td>
      <td>-0.193677</td>
    </tr>
    <tr>
      <th>video_id</th>
      <td>-0.000853</td>
      <td>1.000000</td>
      <td>0.011859</td>
      <td>0.002554</td>
      <td>0.006507</td>
      <td>0.010515</td>
      <td>0.008753</td>
      <td>0.013983</td>
      <td>-0.007083</td>
    </tr>
    <tr>
      <th>video_duration_sec</th>
      <td>-0.011729</td>
      <td>0.011859</td>
      <td>1.000000</td>
      <td>0.013589</td>
      <td>0.004890</td>
      <td>0.002206</td>
      <td>0.003989</td>
      <td>-0.004586</td>
      <td>-0.002981</td>
    </tr>
    <tr>
      <th>video_view_count</th>
      <td>-0.697007</td>
      <td>0.002554</td>
      <td>0.013589</td>
      <td>1.000000</td>
      <td>0.832832</td>
      <td>0.711313</td>
      <td>0.690048</td>
      <td>0.583485</td>
      <td>0.244693</td>
    </tr>
    <tr>
      <th>video_like_count</th>
      <td>-0.581483</td>
      <td>0.006507</td>
      <td>0.004890</td>
      <td>0.832832</td>
      <td>1.000000</td>
      <td>0.850053</td>
      <td>0.828082</td>
      <td>0.706140</td>
      <td>0.202386</td>
    </tr>
    <tr>
      <th>video_share_count</th>
      <td>-0.504015</td>
      <td>0.010515</td>
      <td>0.002206</td>
      <td>0.711313</td>
      <td>0.850053</td>
      <td>1.000000</td>
      <td>0.710117</td>
      <td>0.620182</td>
      <td>0.171651</td>
    </tr>
    <tr>
      <th>video_download_count</th>
      <td>-0.487096</td>
      <td>0.008753</td>
      <td>0.003989</td>
      <td>0.690048</td>
      <td>0.828082</td>
      <td>0.710117</td>
      <td>1.000000</td>
      <td>0.857679</td>
      <td>0.173396</td>
    </tr>
    <tr>
      <th>video_comment_count</th>
      <td>-0.413799</td>
      <td>0.013983</td>
      <td>-0.004586</td>
      <td>0.583485</td>
      <td>0.706140</td>
      <td>0.620182</td>
      <td>0.857679</td>
      <td>1.000000</td>
      <td>0.149750</td>
    </tr>
    <tr>
      <th>text_length</th>
      <td>-0.193677</td>
      <td>-0.007083</td>
      <td>-0.002981</td>
      <td>0.244693</td>
      <td>0.202386</td>
      <td>0.171651</td>
      <td>0.173396</td>
      <td>0.149750</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Visualize a correlation heatmap of the data.


```python
# Create a heatmap to visualize how correlated variables are
plt.figure(figsize = (10,8))
sns.heatmap(data_upsampled.corr(numeric_only = True), annot = True)
plt.show()

```


    
![png](output_52_0.png)
    


One of the model assumptions for logistic regression is no severe multicollinearity among the features. Take this into consideration as you examine the heatmap and choose which features to proceed with.

**Question:** What variables are shown to be correlated in the heatmap?

Rather expectedly, video likes, comments, shares, and downloads are all somewhat correlated with each other. This makes sense, as engagement is likely to be shown via all of these metrics concurrently. Likes are especially highly correlated, so we can remove that variable to reduce multicollinearity.

We will ignore the correlation of video number with each of these metrics, as that is simply due to how the data was originally sorted in the dataframe.

<img src="images/Construct.png" width="100" height="100" align=left>

## **PACE: Construct**

After analysis and deriving variables with close relationships, it is time to begin constructing the model. Consider the questions in your PACE Strategy Document to reflect on the Construct stage.

### **Task 3a. Select variables**

Set your Y and X variables.

Select the outcome variable.


```python
# Select outcome variable
y = data_upsampled['verified_status']
# We are creating a model to predict whether or not a user is verified.

```

Select the features.


```python
# Select features
X = data_upsampled[['video_duration_sec', 'claim_status', 'author_ban_status',
       'video_view_count', 'video_share_count',
       'video_download_count', 'video_comment_count']]


# Display first few rows of features dataframe
X.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>video_duration_sec</th>
      <th>claim_status</th>
      <th>author_ban_status</th>
      <th>video_view_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59</td>
      <td>claim</td>
      <td>under review</td>
      <td>343296.0</td>
      <td>241.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>32</td>
      <td>claim</td>
      <td>active</td>
      <td>140877.0</td>
      <td>19034.0</td>
      <td>1161.0</td>
      <td>684.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>31</td>
      <td>claim</td>
      <td>active</td>
      <td>902185.0</td>
      <td>2858.0</td>
      <td>833.0</td>
      <td>329.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25</td>
      <td>claim</td>
      <td>active</td>
      <td>437506.0</td>
      <td>34812.0</td>
      <td>1234.0</td>
      <td>584.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19</td>
      <td>claim</td>
      <td>active</td>
      <td>56167.0</td>
      <td>4110.0</td>
      <td>547.0</td>
      <td>152.0</td>
    </tr>
  </tbody>
</table>
</div>



### **Task 3b. Train-test split**

Split the data into training and testing sets.


```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)

```

Confirm that the dimensions of the training and testing sets are in alignment.


```python
# Get shape of each training and testing set
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

```

    (26826, 7) (26826,)
    (8942, 7) (8942,)


### **Task 3c. Encode variables**

Check the data types of the features.


```python
# Check data types
X_train.dtypes

```




    video_duration_sec        int64
    claim_status             object
    author_ban_status        object
    video_view_count        float64
    video_share_count       float64
    video_download_count    float64
    video_comment_count     float64
    dtype: object




```python
# Get unique values in `claim_status`
X_train['claim_status'].unique()


```




    array(['opinion', 'claim'], dtype=object)




```python
# Get unique values in `author_ban_status`
X_train['author_ban_status'].unique()

```




    array(['active', 'under review', 'banned'], dtype=object)



As shown above, the `claim_status` and `author_ban_status` features are each of data type `object` currently. In order to work with the implementations of models through `sklearn`, these categorical features will need to be made numeric. One way to do this is through one-hot encoding.

Encode categorical features in the training set using an appropriate method.


```python
# Select the training features that needs to be encoded
X_train_encoding_needed = X_train[['claim_status', 'author_ban_status']]


# Display first few rows
X_train_encoding_needed.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_status</th>
      <th>author_ban_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28075</th>
      <td>opinion</td>
      <td>active</td>
    </tr>
    <tr>
      <th>28358</th>
      <td>opinion</td>
      <td>active</td>
    </tr>
    <tr>
      <th>27086</th>
      <td>claim</td>
      <td>active</td>
    </tr>
    <tr>
      <th>7104</th>
      <td>claim</td>
      <td>active</td>
    </tr>
    <tr>
      <th>25784</th>
      <td>opinion</td>
      <td>under review</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Set up an encoder for one-hot encoding the categorical features
X_encoder = OneHotEncoder(drop = 'first', sparse_output = False)

```


```python
# Fit and transform the training features using the encoder
X_train_encoded = X_encoder.fit_transform(X_train_encoding_needed)

```


```python
# Get feature names from encoder
X_encoder.get_feature_names_out()
```




    array(['claim_status_opinion', 'author_ban_status_banned',
           'author_ban_status_under review'], dtype=object)




```python
# Display first few rows of encoded training features
X_train_encoded

```




    array([[1., 0., 0.],
           [1., 0., 0.],
           [0., 0., 0.],
           ...,
           [1., 0., 0.],
           [0., 1., 0.],
           [1., 0., 0.]])




```python
# Place encoded training features (which is currently an array) into a dataframe
X_train_encoded_df = pd.DataFrame(data = X_train_encoded, columns = X_encoder.get_feature_names_out())


# Display first few rows
X_train_encoded_df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_status_opinion</th>
      <th>author_ban_status_banned</th>
      <th>author_ban_status_under review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Display first few rows of `X_train` with `claim_status` and `author_ban_status` columns dropped (since these features are being transformed to numeric)
X_train_final = X_train.drop(columns = ['claim_status','author_ban_status'])

```


```python
# Concatenate `X_train` and `X_train_encoded_df` to form the final dataframe for training data (`X_train_final`)
# Note: Using `.reset_index(drop=True)` to reset the index in X_train after dropping `claim_status` and `author_ban_status`,
# so that the indices align with those in `X_train_encoded_df` and `count_df`
X_train_final = pd.concat([X_train_final.reset_index(drop = True), X_train_encoded_df], axis = 1)

# Display first few rows
X_train_final.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>video_duration_sec</th>
      <th>video_view_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
      <th>claim_status_opinion</th>
      <th>author_ban_status_banned</th>
      <th>author_ban_status_under review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>52</td>
      <td>7612.0</td>
      <td>105.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>23</td>
      <td>9303.0</td>
      <td>593.0</td>
      <td>49.0</td>
      <td>13.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11</td>
      <td>366577.0</td>
      <td>40248.0</td>
      <td>1486.0</td>
      <td>598.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>43</td>
      <td>176311.0</td>
      <td>22270.0</td>
      <td>1546.0</td>
      <td>557.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24</td>
      <td>4320.0</td>
      <td>194.0</td>
      <td>23.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



Check the data type of the outcome variable.


```python
# Check data type of outcome variable
y_train.dtypes

```




    dtype('O')




```python
# Get unique values of outcome variable
y_train.unique()

```




    array(['verified', 'not verified'], dtype=object)



A shown above, the outcome variable is of data type `object` currently. One-hot encoding can be used to make this variable numeric.

Encode categorical values of the outcome variable the training set using an appropriate method.


```python
# Set up an encoder for one-hot encoding the categorical outcome variable
y_encoder = OneHotEncoder(drop = 'first', sparse_output = False)

```


```python
# Encode the training outcome variable
# Notes:
#   - Adjusting the shape of `y_train` before passing into `.fit_transform()`, since it takes in 2D array
#   - Using `.ravel()` to flatten the array returned by `.fit_transform()`, so that it can be used later to train the model
y_train_final = y_encoder.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# Display the encoded training outcome variable
y_train_final

```




    array([1., 1., 1., ..., 0., 0., 0.])



### **Task 3d. Model building**

Construct a model and fit it to the training set.


```python
# Construct a logistic regression model and fit it to the training set
log_clf = LogisticRegression(random_state = 42, max_iter = 800).fit(X_train_final, y_train_final)

```

<img src="images/Execute.png" width="100" height="100" align=left>

## **PACE: Execute**

Consider the questions in your PACE Strategy Document to reflect on the Execute stage.

### **Taks 4a. Results and evaluation**

Evaluate your model.

Encode categorical features in the testing set using an appropriate method.


```python
# Select the testing features that needs to be encoded
X_test_encoding_needed = X_test[['claim_status', 'author_ban_status']]


# Display first few rows
X_test_encoding_needed.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_status</th>
      <th>author_ban_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>26849</th>
      <td>opinion</td>
      <td>active</td>
    </tr>
    <tr>
      <th>1227</th>
      <td>claim</td>
      <td>under review</td>
    </tr>
    <tr>
      <th>25467</th>
      <td>opinion</td>
      <td>active</td>
    </tr>
    <tr>
      <th>27310</th>
      <td>opinion</td>
      <td>active</td>
    </tr>
    <tr>
      <th>27560</th>
      <td>opinion</td>
      <td>active</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Transform the testing features using the encoder
# from before : X_encoder = OneHotEncoder(drop = 'first', sparse_output = False)
X_test_encoded = X_encoder.fit_transform(X_test_encoding_needed)


# Display first few rows of encoded testing features
X_test_encoded

```




    array([[1., 0., 0.],
           [0., 0., 1.],
           [1., 0., 0.],
           ...,
           [1., 0., 0.],
           [1., 0., 0.],
           [1., 0., 0.]])




```python
# Place encoded testing features (which is currently an array) into a dataframe
X_test_encoded_df = pd.DataFrame(data=X_test_encoded, columns=X_encoder.get_feature_names_out())


# Display first few rows
X_test_encoded_df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_status_opinion</th>
      <th>author_ban_status_banned</th>
      <th>author_ban_status_under review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Display first few rows of `X_test` with `claim_status` and `author_ban_status` columns dropped (since these features are being transformed to numeric)
X_test_final = X_test.drop(columns = ['claim_status','author_ban_status'])
X_test_final.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>video_duration_sec</th>
      <th>video_view_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>26849</th>
      <td>12</td>
      <td>2177.0</td>
      <td>168.0</td>
      <td>6.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1227</th>
      <td>43</td>
      <td>862507.0</td>
      <td>86137.0</td>
      <td>3206.0</td>
      <td>265.0</td>
    </tr>
    <tr>
      <th>25467</th>
      <td>13</td>
      <td>7457.0</td>
      <td>100.0</td>
      <td>7.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>27310</th>
      <td>6</td>
      <td>6388.0</td>
      <td>133.0</td>
      <td>9.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>27560</th>
      <td>34</td>
      <td>4075.0</td>
      <td>325.0</td>
      <td>16.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Concatenate `X_test` and `X_test_encoded_df` to form the final dataframe for training data (`X_test_final`)
# Note: Using `.reset_index(drop=True)` to reset the index in X_test after dropping `claim_status`, and `author_ban_status`,
# so that the indices align with those in `X_test_encoded_df` and `test_count_df`
X_test_final = pd.concat([X_test_final.reset_index(drop = True), X_test_encoded_df], axis = 1)


# Display first few rows
X_test_final.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>video_duration_sec</th>
      <th>video_view_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
      <th>claim_status_opinion</th>
      <th>author_ban_status_banned</th>
      <th>author_ban_status_under review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12</td>
      <td>2177.0</td>
      <td>168.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>43</td>
      <td>862507.0</td>
      <td>86137.0</td>
      <td>3206.0</td>
      <td>265.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13</td>
      <td>7457.0</td>
      <td>100.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>6388.0</td>
      <td>133.0</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>34</td>
      <td>4075.0</td>
      <td>325.0</td>
      <td>16.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Test the logistic regression model. Use the model to make predictions on the encoded testing set.


```python
# Use the logistic regression model to get predictions on the encoded testing set
y_pred = log_clf.predict(X_test_final)

```

Display the predictions on the encoded testing set.


```python
# Display the predictions on the encoded testing set
y_pred

```




    array([1., 0., 1., ..., 1., 1., 1.])



Display the true labels of the testing set.


```python
# Display the true labels of the testing set
y_test

```




    26849        verified
    1227     not verified
    25467        verified
    27310        verified
    27560        verified
                 ...     
    16315    not verified
    19045        verified
    30024        verified
    17501    not verified
    29483        verified
    Name: verified_status, Length: 8942, dtype: object



Encode the true labels of the testing set so it can be compared to the predictions.


```python
# Encode the testing outcome variable
# Notes:
#   - Adjusting the shape of `y_test` before passing into `.transform()`, since it takes in 2D array
#   - Using `.ravel()` to flatten the array returned by `.transform()`, so that it can be used later to compare with predictions
y_test_final = y_encoder.transform(y_test.values.reshape(-1, 1)).ravel()

# Display the encoded testing outcome variable
y_test_final
```




    array([1., 0., 1., ..., 1., 0., 1.])



Confirm again that the dimensions of the training and testing sets are in alignment since additional features were added.


```python
# Get shape of each training and testing set
X_train_final.shape, y_train_final.shape, X_test_final.shape, y_test_final.shape

```




    ((26826, 8), (26826,), (8942, 8), (8942,))



### **Task 4b. Visualize model results**

Create a confusion matrix to visualize the results of the logistic regression model.


```python
# Compute values for confusion matrix
cm = metrics.confusion_matrix(y_test_final, y_pred, labels = log_clf.classes_)

# Create display of confusion matrix
disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = log_clf.classes_)

# Plot confusion matrix
disp.plot()

# Display plot
plt.show()
```


    
![png](output_109_0.png)
    


Create a classification report that includes precision, recall, f1-score, and accuracy metrics to evaluate the performance of the logistic regression model.


```python
# Create a classification report
print('Accuracy: ', "%.6f" % metrics.accuracy_score(y_test_final, y_pred))
print('Precision: ', "%.6f" % metrics.precision_score(y_test_final, y_pred))
print('Recall: ', "%.6f" % metrics.recall_score(y_test_final, y_pred))
print('F1 Score: ', "%.6f" % metrics.f1_score(y_test_final, y_pred))

```

    Accuracy:  0.659025
    Precision:  0.617344
    Recall:  0.850166
    F1 Score:  0.715286


### **Task 4c. Interpret model coefficients**


```python
# Get the feature names from the model and the model coefficients (which represent log-odds ratios)
# Place into a DataFrame for readability
pd.DataFrame(data={"Feature Name":log_clf.feature_names_in_, "Model Coefficient":log_clf.coef_[0]})

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature Name</th>
      <th>Model Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>video_duration_sec</td>
      <td>0.008352</td>
    </tr>
    <tr>
      <th>1</th>
      <td>video_view_count</td>
      <td>-0.000002</td>
    </tr>
    <tr>
      <th>2</th>
      <td>video_share_count</td>
      <td>0.000007</td>
    </tr>
    <tr>
      <th>3</th>
      <td>video_download_count</td>
      <td>-0.000238</td>
    </tr>
    <tr>
      <th>4</th>
      <td>video_comment_count</td>
      <td>0.000416</td>
    </tr>
    <tr>
      <th>5</th>
      <td>claim_status_opinion</td>
      <td>0.000369</td>
    </tr>
    <tr>
      <th>6</th>
      <td>author_ban_status_banned</td>
      <td>-0.000017</td>
    </tr>
    <tr>
      <th>7</th>
      <td>author_ban_status_under review</td>
      <td>0.000001</td>
    </tr>
  </tbody>
</table>
</div>



### **Task 4d. Conclusion**

1. What are the key takeaways from this project?

2. What results can be presented from this project?

==> ENTER YOUR RESPONSE TO QUESTIONS 1 AND 2 HERE

This model is, perhaps, useful, but it is also fraught with error. Accuracy of 66% is not very strong, but the recall of 85% is good. Multicollinearity is definitely an issue with this dataset, so we dropped the video_like_count, as it had the strongest correlation wiht other variables.

The most useful piece of information: every additional second in video length is associated with approximately an increase of 0.008 in the log-odds that an account is verified. 

For stakeholders: A loose extension of this is that verified users tend to make longer videos.



**Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged. 
