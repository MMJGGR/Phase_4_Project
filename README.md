# Phase_4_Project# **City of Chicago Traffic Crashes**
<div style="width: 2000px; overflow: hidden;">
  <header style="height: 5in; display: flex; justify-content: center; align-items: center;">
    <img src="chicago2.png" alt="Chicago skyline" style="height: 5in; width: 100%; object-fit: cover;"/>
  </header>
</div>

# **1. Business Understanding**
### **Problem:**
Traffic accidents are a significant issue in large cities like Chicago, causing property damage, injuries, and fatalities. Identifying the primary causes of these accidents can help city planners, traffic safety boards, and policymakers take proactive measures to reduce accidents and improve road safety. The dataset used in this project, provided by the City of Chicago, includes detailed information about accidents, vehicles, and the people involved, offering a rich resource for understanding the underlying causes of crashes.

### **Stakeholders:**
- **Vehicle Safety Boards**: These organizations are tasked with analyzing traffic accidents and implementing strategies to prevent future incidents. They could use the findings from this project to identify key risk factors contributing to accidents and introduce targeted interventions.
- **City of Chicago**: City officials and traffic planners could benefit from the insights gained from this project by making data-driven decisions to improve road safety infrastructure, optimize traffic management, and reduce accident rates.

### **Project Goals:**
The goal of this project is to build a model that predicts the **primary contributory cause** of a car accident based on factors such as road conditions, vehicle characteristics, and the people involved. Since the original dataset contains over 40 unique contributory causes, we have grouped these into 5 main categories to make the task more manageable. The project also uses different classification models and iterating on them to achieve the best possible results. The target variable, `PRIM_CONTRIBUTORY_CAUSE`, represents the main reason for each accident.

### **Approach:**
- **Target Simplification**: To make the problem manageable, we grouped the 40 unique values of the `PRIM_CONTRIBUTORY_CAUSE` column into 5 broad categories.
- **Classification Models**: We approached this as a classification problem, iterating through various models like Logistic Regression, Random Forest, and XGBoost, and Neural Networks, refining hyperparameters and improving upon each model.


### **Objectives:**

1. **Main Objective**:
   - Build a model to predict the **PRIM_CONTRIBUTORY_CAUSE** of car accidents. The model should highlight the key factors contributing to accidents, such as **driver error**, **environmental factors**, **alcohol/drugs**, **mechanical failures**,  **pedestrian/cyclist errors**, and **others**. These insights will support traffic safety boards in designing targeted prevention strategies.

2. **Data Quality**:
   - Ensure the dataset is of high quality by maintaining completeness and accuracy, especially in critical variables like road conditions, weather, and vehicle information. Reliable data will enable more accurate predictions and ensure robust models.

3. **Data Imbalance**:
   - Address the severe imbalance in the target variable (`PRIM_CONTRIBUTORY_CAUSE`) by applying techniques such as **SMOTE**, **class weighting**, or **ensemble methods**. Handling imbalance effectively will result in models that perform better across all categories, not just the dominant ones.

4. **Feature Importance**:
   - Investigate the relationships between key features, such as **road conditions**, **vehicle types**, and **driver behavior**. Interaction features and deeper insights into how these variables influence accidents can improve model performance and offer more actionable insights for stakeholders.


# **2. Data Understanding**
The dataset used for this project comes from the City of Chicago and includes detailed information on vehicle crashes, along with additional data on the involved vehicles and people. (https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if/about_data)
### **a) Imported relevant modules**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#### **b) Loading the dataset**
from google.colab import drive


# Mount your Google Drive
drive.mount('/content/drive')

# Now you can read the CSV file
data = pd.read_csv('/content/drive/My Drive/Traffic_Crashes_-_Crashes_20241010.csv')  # Updated path
data.head()
#data = pd.read_csv('..\Traffic_Crashes_-_Crashes_20241010.csv')
#data.head()
### **c) Data Shape**
print('Our data has {} rows and {} columns'.format(data.shape[0], data.shape[1]))
### **d) Data Description**
data.describe()
Our data includes conditions surrounding a crash as well as the crash's outcome.

The columns include:
1. CRASH_RECORD_ID: Unique ID for each crash, used to link to related datasets.
2. CRASH_DATE_EST_I: Estimated crash date if reported later.
3. CRASH_DATE: Date and time of the crash.
4. POSTED_SPEED_LIMIT: Speed limit at the crash location.
5. TRAFFIC_CONTROL_DEVICE: Traffic control device present.
6. DEVICE_CONDITION: Condition of the traffic control device.
7. WEATHER_CONDITION: Weather at the time of the crash.
8. LIGHTING_CONDITION: Lighting at the time of the crash.
9. FIRST_CRASH_TYPE: Type of first collision.
10. TRAFFICWAY_TYPE: Type of trafficway.
11. LANE_CNT: Number of through lanes.
12. ALIGNMENT: Street alignment.
13. ROADWAY_SURFACE_COND: Road surface condition.
14. ROAD_DEFECT: Road defects.
15. REPORT_TYPE: Type of report (at scene, at desk, amended).
16. CRASH_TYPE: Severity classification of the crash.
17. INTERSECTION_RELATED_I: Whether an intersection played a role.
18. NOT_RIGHT_OF_WAY_I: Whether the crash occurred outside the public right-of-way.
19. HIT_AND_RUN_I: Whether it was a hit-and-run.
20. DAMAGE: Estimated damage.
21. DATE_POLICE_NOTIFIED: Date police were notified.
22. **PRIM_CONTRIBUTORY_CAUSE: Primary cause of the crash**
23. SEC_CONTRIBUTORY_CAUSE: Secondary cause of the crash.
24. STREET_NO: Street address number.
25. STREET_DIRECTION: Street address direction.
26. STREET_NAME: Street address name.
27. BEAT_OF_OCCURRENCE: Chicago Police Department Beat ID.
28. PHOTOS_TAKEN_I: Whether photos were taken.
29. STATEMENTS_TAKEN_I: Whether statements were taken.
30. DOORING_I: Whether it involved dooring.
31. WORK_ZONE_I: Whether it occurred in a work zone.
32. WORK_ZONE_TYPE: Type of work zone.
33. WORKERS_PRESENT_I: Whether workers were present.
34. NUM_UNITS: Number of units involved.
35. MOST_SEVERE_INJURY: Most severe injury sustained1.
36. INJURIES_TOTAL: Total number of injuries.
37. NJURIES_FATAL: Number of fatal injuries.
38. INJURIES_INCAPACITATING: Number of incapacitating injuries.
39. INJURIES_NON_INCAPACITATING: Number of non-incapacitating injuries.
40. INJURIES_REPORTED_NOT_EVIDENT: Number of reported but not evident injuries.
41. INJURIES_NO_INDICATION: Number of no indication of injuries.
42. INJURIES_UNKNOWN: Number of unknown injuries.
43. CRASH_HOUR: Hour of the crash.
44. CRASH_DAY_OF_WEEK: Day of the week of the crash.
45. CRASH_MONTH: Month of the crash.
46. LATITUDE: Latitude of the crash location.
47. LONGITUDE: Longitude of the crash location.
48. LOCATION: Geographic location of the crash.

PRIM_CONTRIBUTORY_CAUSE and SEC_CONTRIBUTORY_CAUSE are closely related. Below we compare how frequently each variable appears in the respective columns
print('Variabe, PRIM_CONTRIBUTORY_CAUSE, SEC_CONTRIBUTORY_CAUSE') #creates a header for our output

#summarizes variables in our target by percentage
for i in data['PRIM_CONTRIBUTORY_CAUSE'].unique():
    prim_percentage = (data['PRIM_CONTRIBUTORY_CAUSE'].value_counts()[i]/data['PRIM_CONTRIBUTORY_CAUSE'].value_counts().sum())*100
    sec_percentage = (data['SEC_CONTRIBUTORY_CAUSE'].value_counts()[i]/data['SEC_CONTRIBUTORY_CAUSE'].value_counts().sum())*100 if i in data['SEC_CONTRIBUTORY_CAUSE'].unique() else None
    print(f"{i}, {prim_percentage:.2f}%", end='')
    if sec_percentage is not None:
        print(f", {sec_percentage:.2f}%", end='')
    print()
'PRIM_CONTRIBUTORY_CAUSE' and 'SEC_CONTRIBUTORY_CAUSE' are obviously very highly related. Hence we will drop 'SEC_CONTRIBUTORY_CAUSE'

print(f'Our target variable has {data["PRIM_CONTRIBUTORY_CAUSE"].nunique()} unique values')
Below we summarize related variables into 6 main groups based on how closely they are related to reduce the number of unique values and make our classification task easier to handle
def categorize_cause(cause):
    if cause in [
        'FAILING TO YIELD RIGHT-OF-WAY', 'FOLLOWING TOO CLOSELY', 'IMPROPER LANE USAGE',
        'DISREGARDING TRAFFIC SIGNALS', 'DISTRACTION - FROM INSIDE VEHICLE',
        'DISTRACTION - FROM OUTSIDE VEHICLE', 'IMPROPER TURNING/NO SIGNAL',
        'IMPROPER BACKING', 'TURNING RIGHT ON RED', 'DRIVING ON WRONG SIDE/WRONG WAY',
        'DISREGARDING STOP SIGN', 'CELL PHONE USE OTHER THAN TEXTING', 'TEXTING',
        'HAD BEEN DRINKING (USE WHEN ARREST IS NOT MADE)'
    ]:
        return 'Driver Error'

    elif cause in [
        'WEATHER', 'VISION OBSCURED (SIGNS, TREE LIMBS, BUILDINGS, ETC.)', 'ROADWAY SURFACE COND_ICE',
        'ROAD CONSTRUCTION/MAINTENANCE', 'ROAD DEFECT_UNKNOWN', 'LIGHTING_CONDITION_DARKNESS',
        'LIGHTING_CONDITION_DAWN', 'LIGHTING_CONDITION_DUSK'
    ]:
        return 'Environmental Factors'

    elif cause in [
        'EQUIPMENT - VEHICLE CONDITION', 'BRAKE FAILURE', 'TIRE FAILURE', 'ENGINE FAILURE',
        'AIRBAG DEPLOYED_DID NOT DEPLOY'
    ]:
        return 'Mechanical Failures'

    elif cause in [
        'PEDESTRIAN ACTIONS', 'BICYCLE ADVANCING LEGALLY ON RED LIGHT', 'RELATED TO BUS STOP',
        'OBSTRUCTED CROSSWALKS', 'ELECTRONIC DEVICE USE BY PEDESTRIAN'
    ]:
        return 'Pedestrian/Cyclist Errors'

    elif cause in [
        'UNDER THE INFLUENCE OF ALCOHOL/DRUGS (USE WHEN ARREST IS EFFECTED)', 'PHYSICAL CONDITION OF DRIVER',
        'SLEEPING AT THE WHEEL'
    ]:
        return 'Alcohol/Drugs and Physical Condition'

    else:
        return 'Other'

data['PRIM_CONTRIBUTORY_CAUSE_GROUPED'] = data['PRIM_CONTRIBUTORY_CAUSE'].apply(categorize_cause)
data = data.drop(columns=['PRIM_CONTRIBUTORY_CAUSE'])

print(f'Our target variable now has {data["PRIM_CONTRIBUTORY_CAUSE_GROUPED"].nunique()} unique values')

# Compare the distribution of variables in our target
plt.figure(figsize=(18, 8))

sns.histplot(data['PRIM_CONTRIBUTORY_CAUSE_GROUPED'], color='orange', label='Distributions', stat='density', kde=True)

plt.legend()
plt.title('PRIM_CONTRIBUTORY_CAUSE_GROUPED Distribution')
plt.show()

### **e) Duplicates**
data.duplicated().sum()
There are no duplicates in our dataset
### **f) Datatypes**
data.info()
Our dataset is quite large with several columns that seem to contain similar information.



Below we drop some columns that have limited useful information given our overall objective and similarity to other columns which we will include in our dataset.
columns_to_drop = ['CRASH_RECORD_ID','CRASH_DATE_EST_I', 'CRASH_DATE','REPORT_TYPE', 'PHOTOS_TAKEN_I','STATEMENTS_TAKEN_I',
'DOORING_I','STREET_NO','STREET_DIRECTION','DATE_POLICE_NOTIFIED','BEAT_OF_OCCURRENCE','LANE_CNT','LATITUDE',
'LONGITUDE','INJURIES_REPORTED_NOT_EVIDENT','INJURIES_NO_INDICATION','INJURIES_UNKNOWN','SEC_CONTRIBUTORY_CAUSE', 'LOCATION']

relevant_data = data.drop(columns = columns_to_drop, axis=1)
relevant_data.head()
print('Our data now has {} rows and {} columns'.format(relevant_data.shape[0], relevant_data.shape[1]))
### **g) Missing Values**
Next, we will look at missing data by column percentage.
relevant_data.isna().sum()/data.shape[0]*100
We will start by inspecting and handling columns with a large amount of missing data
for col in relevant_data.columns:
    if relevant_data[col].isna().sum()/relevant_data.shape[0]*100 > 50:
        print(f'{col}\n Missing value percent {relevant_data[col].isna().sum()/relevant_data.shape[0]*100:.2f}\n Unique values {relevant_data[col].unique()}')
We can drop these columns as they contain a large amount of missing data and imputing them based on mean, mode or median will change their distributions
columns_to_drop = [col for col in relevant_data.columns if relevant_data[col].isna().sum()/relevant_data.shape[0]*100 > 50]
relevant_data = relevant_data.drop(columns = columns_to_drop, axis=1)

print(relevant_data.isna().sum())

We can drop the remaining columns with missing values as their number is small and unlikely to affect the overall data distribution materially
relevant_data = relevant_data.dropna()
print('We now have {} rows with missing values'.format(relevant_data.isna().sum().sum()))
print('Our data now has {} rows and {} columns'.format(relevant_data.shape[0], relevant_data.shape[1]))
### **h) Variable Types**
# Separate categorical columns
categorical_columns = relevant_data.select_dtypes(include=['object']).columns.tolist()

# Separate continuous (numerical) columns
continuous_columns = relevant_data.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Display the separated columns
print(f"We have {len(categorical_columns)} categorical columns and {len(continuous_columns)} continuous columns")
# **3. EDA & Data Preparation**
### **a) Basic Descriptive Statistics**
- **Continuous Columns:** We will get an overview of the distribution, central tendency, and spread
relevant_data[continuous_columns].describe()
Our continuous data seems to have different scales across different features. Hence we may need to scale it
- **Categorical Columns:** We will get an overview of the distribution in each categorical column
relevant_data[categorical_columns].describe()
Some of our categorical features have a lot of categories. Hence we may need to use **target encoding** to deal with high cardinality.

Target encoding replaces each category with the mean of the target variable for that category. This reduces the dimensionality by not increasing the number of features, which can help in preventing overfitting. Example: If a category is frequently associated with fatalities, the encoding will reflect that association. Target encoding should not alter the shape of our data.
### **c) Visualizations**
We will use visualizations to help display the relationships and patterns in our data intuitively.

- **Distribution of continuous features:**

**1) Histograms**
import matplotlib.pyplot as plt
relevant_data[continuous_columns].hist(figsize=(10, 10), bins=10) #plot histograms
plt.show()

Although we currently classify 'CRASH_HOUR', 'CRASH_DAY_OF_WEEK', 'CRASH_MONTH', as continuous features, their distributions imply that they represent discrete time periods rather than a continuous range. Each hour, day, and month is a distinct category which may have unique traffic patterns. We will convert these features to categorical
time_col = ['CRASH_HOUR', 'CRASH_DAY_OF_WEEK', 'CRASH_MONTH']


for col in time_col:
  if col in relevant_data[continuous_columns]:
    categorical_columns.append(col)
    continuous_columns.remove(col)
    relevant_data[col] = relevant_data[col].astype('object')
  else:
    continue
print('Time columns already removed')


print(f"We now have {len(categorical_columns)} categorical columns and {len(continuous_columns)} continuous columns")
We can also visualize our continuous columns using boxplots to check for outliers

**2) Boxplots**
# Create a grid of subplots with 2 rows and 3 columns
n_cols = 3
n_rows = (len(continuous_columns) + n_cols - 1) // n_cols  # This ensures enough rows

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))  # create subplots

# Iterate over the continuous columns and plot each one in a separate subplot
for i, col in enumerate(continuous_columns):
    # Get the appropriate subplot
    ax = axes[i // n_cols, i % n_cols]

    sns.boxplot(y=relevant_data[col], ax=ax)

    # Set the title of the subplot
    ax.set_title(col)

    # Set labels for y-axis
    ax.set_ylabel("Values")

# Hide any empty subplots
for i in range(len(continuous_columns), n_rows * n_cols):
    ax = axes[i // n_cols, i % n_cols]
    ax.axis('off')

# Adjust the layout to prevent overlapping
plt.tight_layout()

# Show the plot
plt.show()
Looking at the boxplots, our continuous columns appear concentrated in certain values likely due to the nature of the crash data. It seems like there might be a lot of zero values which might give the appearance of categories but the features are actually continuous.

- **POSTED_SPEED_LIMIT:** This is continuous, but there might be common values (e.g., 30, 40 mph) that make it appear categorical.
- **NUM_UNITS:** The number of units (vehicles, people) involved in a crash. This is likely continuous (discrete, but not categorical).
- **INJURIES_TOTAL, INJURIES_FATAL, INJURIES_INCAPACITATING, INJURIES_NON_INCAPACITATING:** These are continuous as they count the number of injuries, though they might have many zeroes.

Further, our data does not seem to have any outliers.
**3) Bar Plots**
#we will remove STREET_NAME as it is hard to represent in the plot
if 'STREET_NAME' in categorical_columns:
  categorical_columns.remove('STREET_NAME')
else:
  print('STREET_NAME column already removed')

# Determine number of rows and columns for the subplots based on the number of categorical columns
n_cols = 4
n_rows = (len(categorical_columns[:8]) + n_cols - 1) // n_cols  # This ensures enough rows

fig, axes = plt.subplots(n_rows, n_cols,figsize=(25, 12)) #create subplots

axes = axes.flatten() #flatten axes

#plot each categorical column in a subplot
for i, col in enumerate(categorical_columns[:8]):
    relevant_data[col].value_counts().plot(kind = 'bar', ax=axes[i], color = 'skyblue')
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')

#remove empty subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout() #prevent overlapping
plt.show()
for col in categorical_columns:
    print(col, relevant_data[col].nunique())
#plot the rest of the categorical columns
#we will remove STREET_NAME as it is hard to represent in the plot
if 'STREET_NAME' in categorical_columns:
  categorical_columns.remove('STREET_NAME')
else:
  print('STREET_NAME column already removed')

# Determine number of rows and columns for the subplots based on the number of categorical columns
n_cols = 4
n_rows = (len(categorical_columns[8:]) + n_cols - 1) // n_cols  # This ensures enough rows

fig, axes = plt.subplots(n_rows, n_cols,figsize=(25, 12)) #create subplots

axes = axes.flatten() #flatten axes

#plot each categorical column in a subplot
for i, col in enumerate(categorical_columns[8:]):
    relevant_data[col].value_counts().plot(kind = 'bar', ax=axes[i], color = 'skyblue')
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')

#remove empty subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout() #prevent overlapping
plt.show()
print(f"We now have {len(categorical_columns)} categorical columns and {len(continuous_columns)} continuous columns")
**Specific Observations for Data Understanding and Preparation:**

**Target Variable (`PRIM_CONTRIBUTORY_CAUSE_GROUPED`)**
- **Highly imbalanced**
- **High cardinality** (6 categories)
- **Approach**: Consider label encoding.

**High Cardinality Features**
- `TRAFFIC_CONTROL_DEVICE`, `FIRST_CRASH_TYPE`, `TRAFFICWAY_TYPE`,`STREET_NAME`
- **Approach**: Apply target\frequency encoding technique.

**Severely Imbalanced Features**
- `WEATHER_CONDITION`, `LIGHTING_CONDITION`, `ROAD_DEFECT`, `DAMAGE`, `MOST_SEVERE_INJURY`
- **Approach**: Group rare categories or use resampling techniques.

**Temporal Features**
- `CRASH_HOUR`, `CRASH_DAY_OF_WEEK`, `CRASH_MONTH`
- **Approach**: Create bins.

**Related Variables**
- `DEVICE_CONDITION` and `TRAFFIC_CONTROL_DEVICE`
- `WEATHER_CONDITION` and `ROADWAY_SURFACE_COND`
- **Approach**: Create interaction features or combined categories.



**Modeling Guidelines**

**Handling Imbalance**
- Use **SMOTE**, **class weighting**, or **ensemble methods**.

**Feature Engineering**
- Create **interaction terms**.
- Bin **continuous variables**.
- Encode **cyclical features**.

**Model Selection**
- **Tree-based models**: Random Forest, Gradient Boosting.
- **Neural networks** with entity embeddings.

**Evaluation Metrics**
- **F1-score**, **precision**, **recall**, **Accuracy**.

**Cross-validation**
- Use **stratified k-fold cross-validation**.

**Interpretability**
- Use **SHAP values** or **LIME** for model explanations.

### **d) Correlation Analysis**
plt.subplots(figsize=(20,15))
sns.heatmap(relevant_data[continuous_columns].corr(),cmap="coolwarm",annot=True);
Our correlation analysis only includes numerical features with only INJURIES_TOTAL and INJURY_NON_INCAPACITATING having high correlation (77%) which is to be expected.
### **e) Encoding, Interaction Terms and Binning**

We will start by splitting our data into a training and test set before proceeding to avoid any data leakage

### **Train Test Split**
from sklearn.model_selection import train_test_split

X=relevant_data.drop('PRIM_CONTRIBUTORY_CAUSE_GROUPED', axis=1)
y=relevant_data['PRIM_CONTRIBUTORY_CAUSE_GROUPED']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
### **Binning POSTED_SPEED_LIMIT, CRASH_HOUR, INJURIES_TOTAL**

Binning  involves converting continuous variables into categorical ones by grouping them into defined intervals or "bins. Binning helps in improved model performance as it reduces noise, handles outliers, simplifies relationships, and increases interpretability.
# Define bins and labels for POSTED_SPEED_LIMIT
speed_bins = [0, 20, 40, 60, float('inf')]
speed_labels = ['Low Speed (0-20)', 'Moderate Speed (21-40)', 'High Speed (41-60)', 'Very High Speed (61+)']

# Define bins and labels for CRASH_HOUR
hour_bins = [0, 6, 12, 18, 24]
hour_labels = ['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)']

# Define bins and labels for INJURIES_TOTAL
injury_bins = [0, 1, 4, 7, float('inf')]
injury_labels = ['No Injuries', 'Few Injuries (1-3)', 'Moderate Injuries (4-6)', 'High Injuries (7+)']

# Apply binning to the training set
X_train['speed_limit_binned'] = pd.cut(X_train['POSTED_SPEED_LIMIT'], bins=speed_bins, labels=speed_labels, right=False)
X_train['crash_hour_binned'] = pd.cut(X_train['CRASH_HOUR'], bins=hour_bins, labels=hour_labels, right=False)
X_train['injuries_binned'] = pd.cut(X_train['INJURIES_TOTAL'], bins=injury_bins, labels=injury_labels, right=False)

# Apply binning to the test set
X_test['speed_limit_binned'] = pd.cut(X_test['POSTED_SPEED_LIMIT'], bins=speed_bins, labels=speed_labels, right=False)
X_test['crash_hour_binned'] = pd.cut(X_test['CRASH_HOUR'], bins=hour_bins, labels=hour_labels, right=False)
X_test['injuries_binned'] = pd.cut(X_test['INJURIES_TOTAL'], bins=injury_bins, labels=injury_labels, right=False)

### **Interaction Terms FOR WEATHER_CONDITION and ROADWAY_SURFACE_COND**

Interaction terms capture the combined effect of two or more features on the target variable, revealing relationships that might not be apparent when examining each feature individually. This will improve model performance by allowing the models to learn more complex patterns and dependencies in the data. This will help improve predictive accuracy and offers deeper insights into the underlying factors influencing the outcome.

from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
le_weather = LabelEncoder()
le_roadway = LabelEncoder()

# Fit and transform the training set
X_train['WEATHER_CONDITION_encoded'] = le_weather.fit_transform(X_train['WEATHER_CONDITION'])
X_train['ROADWAY_SURFACE_COND_encoded'] = le_roadway.fit_transform(X_train['ROADWAY_SURFACE_COND'])

# For the test set, only transform
X_test['WEATHER_CONDITION_encoded'] = le_weather.transform(X_test['WEATHER_CONDITION'])
X_test['ROADWAY_SURFACE_COND_encoded'] = le_roadway.transform(X_test['ROADWAY_SURFACE_COND'])

# Create interaction terms using the encoded columns
X_train['weather_road_interaction'] = X_train['WEATHER_CONDITION_encoded'] * X_train['ROADWAY_SURFACE_COND_encoded']
X_test['weather_road_interaction'] = X_test['WEATHER_CONDITION_encoded'] * X_test['ROADWAY_SURFACE_COND_encoded']

### **Frequency Encoding for STREET_NAME**

In a classification setting, the frequency of occurrences of each street name can be significant. Some streets may be more prone to accidents or certain types of accidents than others, and encoding these frequencies allows the model to learn from this information. Frequency encoding converts categorical street names into numerical values based on their occurrence counts, making them suitable for various classifiers.Unlike label encoding, which assigns arbitrary integers to categories, frequency encoding reflects the actual count of occurrences. This avoids the risk of introducing a false ordinal relationship among street names.
# Count occurrences of each street name in the training set
street_counts_train = X_train['STREET_NAME'].value_counts()
street_counts_test = X_test['STREET_NAME'].value_counts()

# Map the counts back to the original DataFrame for the training set
X_train['street_name_frequency'] = X_train['STREET_NAME'].map(street_counts_train)

# Apply the same frequency mapping to the test set
X_test['street_name_frequency'] = X_test['STREET_NAME'].map(street_counts_test)

# Drop original features that have been processed
columns_to_drop = [
    'POSTED_SPEED_LIMIT',    # Dropped after binning
    'CRASH_HOUR',            # Dropped after binning
    'INJURIES_TOTAL',        # Dropped after binning
    'STREET_NAME'            # Dropped after encoding
]

# For X_train and X_test, drop the original features
X_train = X_train.drop(columns=columns_to_drop)
X_test = X_test.drop(columns=columns_to_drop)

Next we do encode the rest of our data. First we need to split this into high cardinality and low cardinality based on the number of features
print('Summary of Frequency Distribution in our Features\n')

# Adjusted condition to include both 'object' and 'category' types
high_cardinality_cols = [
    column for column in X_train.columns
    if X_train[column].nunique() > 7 and
    (X_train[column].dtype == 'object' or X_train[column].dtype == 'category' or X_train[column].dtype == 'int')
]

print('High Cardinality Cols:', high_cardinality_cols)

low_cardinality_cols = [
    column for column in X_train.columns
    if X_train[column].nunique() <= 7 and
    (X_train[column].dtype == 'object' or X_train[column].dtype == 'category' or X_train[column].dtype == 'int')
]

print('Low Cardinality Cols:', low_cardinality_cols)

for col in X_train.columns:
    print(col, X_train[col].nunique())
Feature Summary (Before Encoding)

The following features are included in the dataset before applying target encoding and one-hot encoding:

### High Frequency Features (Target Encoding)
- **`TRAFFIC_CONTROL_DEVICE`**: 19 unique values
- **`DEVICE_CONDITION`**: 8 unique values
- **`WEATHER_CONDITION`**: 12 unique values
- **`FIRST_CRASH_TYPE`**: 18 unique values
- **`TRAFFICWAY_TYPE`**: 20 unique values
- **`ALIGNMENT`**: 6 unique values
- **`ROADWAY_SURFACE_COND`**: 7 unique values
- **`ROAD_DEFECT`**: 7 unique values
- **`NUM_UNITS`**: 16 unique values
- **`MOST_SEVERE_INJURY`**: 5 unique values
- **`INJURIES_FATAL`**: 4 unique values
- **`INJURIES_INCAPACITATING`**: 9 unique values
- **`INJURIES_NON_INCAPACITATING`**: 19 unique values
- **`CRASH_DAY_OF_WEEK`**: 7 unique values
- **`CRASH_MONTH`**: 12 unique values

### Low Frequency Features (One-Hot Encoding)
- **`LIGHTING_CONDITION`**: 6 unique values
- **`CRASH_TYPE`**: 2 unique values
- **`DAMAGE`**: 3 unique values

### Binned Features
- **`speed_limit_binned`**: 4 unique values
- **`crash_hour_binned`**: 4 unique values
- **`injuries_binned`**: 4 unique values

### Label/Frequency Encoded Features
- **`WEATHER_CONDITION_encoded`**: 12 unique values
- **`ROADWAY_SURFACE_COND_encoded`**: 7 unique values
- **`street_name_frequency`**: 539 unique values

### Interaction Terms
- **`weather_road_interaction`**: 38 unique values

This feature set will undergo target encoding for high-frequency features and one-hot encoding for low-frequency features to prepare it for the modeling phase.

### **Label Encoding**

Our target 'PRIM_CONTRIBUTORY_CAUSE' has 40 unique values whuch have a natural order so we will use label encoding for this.
Label encoding the target variable is crucial because many machine learning algorithms require numeric input. By converting the categorical target into numerical values, we enable algorithms  to interpret and process the target effectively. This step ensures the model can measure relationships between features and target classes accurately, improving the overall performance and interpretability of the model.
from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder for the target
le = LabelEncoder()

# Reset the index of y_train and y_test
#y_train = y_train.reset_index(drop=True)
#y_test = y_test.reset_index(drop=True)

# Encode y_train and y_test
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Create new Pandas Series with the encoded values and the old indexes
y_train = pd.Series(y_train_encoded, index=y_train.index)
y_test = pd.Series(y_test_encoded, index=y_test.index)

y_train.unique()
y_train.value_counts()
#### **Target Encoding**
We will use target encoding to deal with our high cardinality features.

Target encoding replaces each category with the mean of the target variable for that category. This reduces the dimensionality by not increasing the number of features, which can help in preventing overfitting. Example: If a category is frequently associated with an outcome related to the values in our target (e.g., No INDICATION OF INJURY), the encoding will reflect that association. Target encoding should not alter the shape of our data
!pip install category_encoders
from category_encoders import TargetEncoder

#encode the high cardinality features
te = TargetEncoder(cols=high_cardinality_cols) #create encoder instance
te.fit(X_train, y_train_encoded) #fit the encoder
X_train_te = te.transform(X_train) #transfrom X_train
X_test_te = te.transform(X_test)  #transfrom X_test

#assign the transform values into a dataframe
X_train_te_df = pd.DataFrame(X_train_te, columns=X_train[high_cardinality_cols].columns)
X_test_te_df = pd.DataFrame(X_test_te, columns=X_test[high_cardinality_cols].columns)

#drop the original low cardinality features from our train and test set
X_train = X_train.drop(columns=high_cardinality_cols)
X_test = X_test.drop(columns=high_cardinality_cols)

# reset index of Train and Test
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

#reset index of te df
X_train_te_df = X_train_te_df.reset_index(drop=True)
X_test_te_df = X_test_te_df.reset_index(drop=True)

#update train and test feature set with the encoded values
X_train = pd.concat([X_train, X_train_te_df], axis=1)
X_test = pd.concat([X_test, X_test_te_df], axis=1)
Below we do some tests to check the integrity of our data after target encoding it


X_train.shape
X_train.head()
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

### **One Hot Encoding**
One-hot encoding is essential for converting categorical features into a binary matrix representation, where each category is represented as a separate binary column. This process prevents the model from assuming any ordinal relationship between categorical values, which can be misleading when categories have no inherent order (e.g., weather conditions or traffic control devices). By applying one-hot encoding, we ensure that the model treats each category independently, improving its ability to capture the true relationships between features and the target.
We expect one hot encoding to change the shape of our data by increasing the number of features.
from sklearn.preprocessing import OneHotEncoder

#fit the ohe
ohe = OneHotEncoder(handle_unknown="ignore", drop="first")
ohe.fit(X_train[low_cardinality_cols])

#transform our train and test feature set
X_train_ohe = ohe.transform(X_train[low_cardinality_cols])
X_test_ohe = ohe.transform(X_test[low_cardinality_cols])

#assign the transform values into a dataframe
X_train_ohe_df = pd.DataFrame(X_train_ohe.toarray(), columns=ohe.get_feature_names_out(X_train[low_cardinality_cols].columns))
X_test_ohe_df = pd.DataFrame(X_test_ohe.toarray(), columns=ohe.get_feature_names_out(X_test[low_cardinality_cols].columns))

#drop the original low cardinality features from our train and test set
X_train = X_train.drop(low_cardinality_cols, axis=1)
X_test = X_test.drop(low_cardinality_cols, axis=1)

# reset index of Train and Test
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

#reset index of ohe
X_train_ohe_df = X_train_ohe_df.reset_index(drop=True)
X_test_ohe_df = X_test_ohe_df.reset_index(drop=True)


#update train and test feature set with the encoded values
X_train = pd.concat([X_train, X_train_ohe_df], axis=1 )
X_test = pd.concat([X_test, X_test_ohe_df], axis=1)

X_train.head()
X_test.head()
print('No. of rows in encoded data:', (X_train.shape[0]+X_test.shape[0]))
print('No. of rows in original data:', (relevant_data.shape[0]))
print(f'No of features in encoded data {X_train.shape[1]} in Train Set and {X_test.shape[1]} in test set:')
print('No. of features in original data:', (relevant_data.drop('MOST_SEVERE_INJURY', axis=1).shape[1]))
Our data looks ok.

### **f) Correlation Analysis**

Below, we run correlation analysis afresh after encoding our data as this will give us a better representation of the correlation our features have with the target variable since we have a numerical representation of our dataset.
def analyze_and_plot_correlations(df, threshold=0.5, target=None):
    # Compute correlation matrix
    corr_matrix = df.corr()

    # Plot heatmap
    plt.figure(figsize=(20, 15))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', square=True)
    plt.title('Correlation Matrix of All Features', fontsize=16)
    plt.tight_layout()
    plt.show()

    # Get high correlations
    high_corrs = corr_matrix.unstack()
    high_corrs = high_corrs[(abs(high_corrs) > threshold) & (abs(high_corrs) < 1.0)]
    high_corrs = high_corrs.sort_values(ascending=False).reset_index()
    high_corrs.columns = ['Feature 1', 'Feature 2', 'Correlation']

    # Print highly correlated feature pairs
    print("Highly correlated feature pairs:")
    for _, row in high_corrs.iterrows():
        print(f"{row['Feature 1']} - {row['Feature 2']}: {row['Correlation']:.2f}")

    # Focus on specific feature correlations if provided
    if target is not None:
        target_corrs = corr_matrix[target]
        target_corrs = target_corrs[(abs(target_corrs) > threshold) & (abs(target_corrs) < 1.0)]
        target_corrs = target_corrs.sort_values(key=abs, ascending=False)

        print(f"\nFeatures highly correlated with {target}:")
        for feat, corr in target_corrs.items():
            print(f"{feat}: {corr:.2f}")


X_train_plus_target = pd.concat([X_train, y_train.rename('PRIM_CONTRIBUTORY_CAUSE_GROUPED')], axis=1) #add back y_train to X_train to perform correlation analysis

analyze_and_plot_correlations(X_train_plus_target, threshold=0.3, target='PRIM_CONTRIBUTORY_CAUSE_GROUPED')

Our data is ready for modeling. We have several highly correlated features which is expected after One Hot Encoding. We will keep the unrelated features in mind when we are modeling to prevent the effects of multicollinearity.E.g. we may neeed to use Principal Component Analysis to transform the feature space into a smaller set of uncorrelated variable which will also help in reducing dimensionality.
# 4. Modeling
#import necessary packages
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from keras import models, layers, optimizers, callbacks
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, silhouette_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA

### **1) Baseline Model: Dummy Classifier**
# Fit the DummyClassifier (baseline model)
dummy_clf = DummyClassifier(strategy='most_frequent', random_state=42)
dummy_clf.fit(X_train, y_train)

# Predict using the baseline model on the test set
y_pred_baseline = dummy_clf.predict(X_test)

# Calculate accuracy score
accuracy_baseline = accuracy_score(y_test, y_pred_baseline)
print(f"Baseline Model Accuracy: {accuracy_baseline:.4f}")

# Generate the classification report
print("\nBaseline Model Classification Report:")
print(classification_report(y_test, y_pred_baseline, zero_division=0, target_names=relevant_data['PRIM_CONTRIBUTORY_CAUSE_GROUPED'].unique()));
**Baseline Model Evaluation**

**Baseline Model Accuracy:** The accuracy of the baseline model is **0.5953**, indicating that the model correctly predicts approximately 59.53% of the instances.

**Findings**
- The model exhibits a strong **imbalance in predictions**, as it only predicts the class **"Pedestrian/Cyclist Errors"** effectively, achieving a recall of **1.00**. This indicates that all instances of this class were correctly identified.
- However, the model fails to predict any instances of the other classes, resulting in **zero precision and recall** for them. This highlights a significant shortfall in the model's ability to generalize across all categories.
- The **macro average** metrics indicate overall poor performance, with an F1-score of only **0.12**, emphasizing the need for improvement.


Given the limitations of the baseline model, particularly its inability to predict multiple classes effectively, we will now implement a **Logistic Regression** model. This model is expected to provide a better understanding of the relationships between features and the target variable, potentially improving classification accuracy across all classes. Additionally, logistic regression will allow us to investigate the influence of different factors on the probability of each contributory cause, which is crucial for gaining actionable insights.

Next, we will focus on tuning the logistic regression model to address the issues identified in the baseline evaluation and enhance its predictive capabilities.

### **2) Logistic Regression**

The performance of the baseline dummy classifier highlights the need for a more robust modeling approach, such as Logistic Regression, which is capable of learning from the feature set and identifying patterns that the dummy model failed to capture.
**Data Scaling**
Before fitting the logistic regression model, we applied **data scaling** to our features. Scaling is crucial for the following reasons:
- **Uniformity**: Many machine learning algorithms, including logistic regression, are sensitive to the scale of the input features. Features with larger ranges can disproportionately influence the model's performance, leading to biased results.
- **Improved Convergence**: Scaling helps gradient-based optimization algorithms converge faster and more reliably, which is especially important in models like logistic regression.

**Principal Component Analysis (PCA)**
After scaling the data, we performed **Principal Component Analysis (PCA)** to reduce the dimensionality of our feature space. PCA is necessary for several reasons:
- **Dimensionality Reduction**: With a large number of features, PCA helps simplify the dataset by transforming it into a smaller set of uncorrelated components. This reduces the risk of overfitting while retaining most of the variance in the data.
- **Noise Reduction**: PCA can help filter out noise and less informative features, leading to more robust model performance.
- **Improved Interpretability**: By reducing the number of features, PCA makes it easier to visualize and interpret the underlying patterns in the data, aiding in decision-making.


By scaling our data and applying PCA, we enhance the performance and interpretability of the logistic regression model, ensuring it is more efficient and effective at capturing the relationships within the data. This step is essential given the high dimensionality and varying scales of the features present in our dataset.

scaler = StandardScaler() #scale our featureset
X_train_pca = scaler.fit_transform(X_train)
X_test_pca = scaler.transform(X_test)

pca = PCA(n_components=0.9, svd_solver='full')  # Retain 90% of variance
X_train_pca = pca.fit_transform(X_train_pca)
X_test_pca = pca.transform(X_test_pca)

logreg = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', random_state=42) #instaniate our logistic regression model

logreg.fit(X_train_pca, y_train)

y_pred = logreg.predict(X_test_pca)

# Calculate accuracy score
accuracy_baseline = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Model Accuracy: {accuracy_baseline:.4f}")

# Generate the classification report
print("\nLogistic Regression Model Classification Report:")
print(classification_report(y_test, y_pred_baseline, zero_division=0, target_names=relevant_data['PRIM_CONTRIBUTORY_CAUSE_GROUPED'].unique()));


**Logistic Regression Model Performance :**
- **Accuracy Score**: 0.6303
- **Overall F1_Score**: 0.60


### Summary of Improvements
- The logistic regression model shows a slight improvement in accuracy to **0.6254** compared to the baseline model. However, it still primarily predicts instances of **Pedestrian/Cyclist Errors** effectively while failing to predict the other classes.
- The precision and recall for most classes remain at **0.00**, indicating that the model still struggles to capture the nuances of other contributory causes, similar to the baseline model.

### Next Steps: Addressing Class Imbalance with SMOTE
To tackle the persistent class imbalance that affects both models, we will implement **SMOTE (Synthetic Minority Over-sampling Technique)**. This approach generates synthetic samples for underrepresented classes, enhancing the model's ability to learn from a more balanced dataset. By doing so, we aim to improve the performance of our logistic regression model and achieve better classification results across all contributory causes.


from imblearn.over_sampling import SMOTE


# Apply SMOTE to oversample the minority classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_pca, y_train)



# Fit the logistic regression model on resampled data
log_reg_smote = LogisticRegression(max_iter=10000, solver='lbfgs', penalty ='l2',C=1)
log_reg_smote.fit(X_train_resampled, y_train_resampled)

# Predict on the test set
y_pred_smote = log_reg_smote.predict(X_test_pca)



# Print classification report
print("Logistic Regression Model after SMOTE:")
print('Accuracy score', accuracy_score(y_test, y_pred_smote))
print(classification_report(y_test, y_pred_smote))
**Logistic Regression Before and After SMOTE**


**Logistic Regression Model After SMOTE**
- **Accuracy**: 0.3066


**Summary of Changes**
- **Accuracy**: After applying SMOTE, the accuracy of the logistic regression model dropped significantly to **0.3066**. This indicates a deterioration in the model's overall predictive performance.

- **Class-Specific Performance**:
  - Precision and recall for the **Driver Error** and **Mechanical Failures** classes remain low, suggesting that the model struggles to predict these classes accurately.
  - There is a notable improvement in precision for the **Other** class, indicating that SMOTE has helped the model better identify instances of this class.
  - However, the model's ability to predict **Pedestrian/Cyclist Errors** significantly worsened, with a recall dropping to **0.21**.

**Next Steps: Exploring Random Forests**

Given the drawbacks observed in the logistic regression model after SMOTE, it may be beneficial to explore **Random Forests** as an alternative modeling approach. Random Forests can offer several advantages:
- **Handling Class Imbalance**: Random Forests are generally more robust to class imbalance and can handle categorical features without requiring extensive preprocessing.
- **Feature Importance**: The model provides insights into feature importance, helping to understand which factors contribute most to the predictions.
- **Better Generalization**: Ensemble methods like Random Forests often provide better generalization performance, especially when dealing with complex datasets.

By implementing Random Forests, we hope to achieve more balanced predictions across all classes, enhancing the overall performance of our model.

# Initialize the Random Forest classifier with balanced class weights
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')

# Fit the model on the training data
rf_model.fit(X_train_pca, y_train)
# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test_pca)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)

class_report_rf = classification_report(y_test, y_pred_rf,zero_division=0, target_names=relevant_data['PRIM_CONTRIBUTORY_CAUSE_GROUPED'].unique())

print(f'Random Forest Accuracy: {accuracy_rf:.4f}')
print('Classification Report:')
print(class_report_rf)

# Extract PCA components
pca_components = pca.components_

# Multiply the feature importance by the PCA components
pc_importance = rf_model.feature_importances_[:, np.newaxis]  # Convert to column vector

# Calculate the contributions of the original features by multiplying PCs importance with PCA components
original_feature_contributions = np.dot(pca_components.T, pc_importance).flatten()

# Create a DataFrame to store the original features and their contributions
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,  # Original feature names
    'Importance': original_feature_contributions
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the top 10 most important original features
print(feature_importance_df.head(10))

# Plot the feature importance for the original features
plt.figure(figsize=(20, 10))
plt.barh(feature_importance_df['Feature'].head(10), feature_importance_df['Importance'].head(10))
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top 10 Features by Importance Mapped from PCA')
plt.show()

**Model Comparison: Random Forest vs. Logistic Regression**

**Random Forest Performance**

- **Accuracy**: The Random Forest model achieved a noticeable improvement in accuracy compared to the logistic regression model, with the **63.25%** showing significant enhancement over both the baseline logistic regression and the logistic regression model after SMOTE.
- **Class Distribution**: The model demonstrated better handling of class imbalance, especially with the balanced class weights setting. The precision, recall, and F1-scores improved for several underrepresented classes, indicating that the Random Forest is better equipped to capture more nuanced patterns in the data.
- **Feature Importance**: One of the key advantages of Random Forests is the ability to provide insights into **feature importance**. This allows us to identify which features (e.g., road conditions, vehicle characteristics) are contributing the most to predictions, which is valuable for interpretability.

**Comparison with Logistic Regression Models**
- **Improvement in Class Predictions**: Compared to the logistic regression models (both with and without SMOTE), the Random Forest model significantly improved the predictive performance across most classes. This is especially true for underrepresented classes, where logistic regression struggled.
- **Handling Class Imbalance**: The logistic regression model, even with SMOTE, faced challenges in effectively predicting minority classes. In contrast, the Random Forest model's use of balanced class weights helped mitigate some of the imbalance issues, leading to a more balanced prediction across all classes.
- **Drawbacks**: Although Random Forests showed better performance overall, it may still face limitations in perfectly addressing class imbalance. Additionally, Random Forests can be computationally more expensive and harder to interpret compared to simpler models like logistic regression.

**Summary of Feature Importance (Post PCA)**

**Key Insights:**
- **Road Defects and Surface Conditions**: Several features related to road conditions (e.g., surface defects like worn surfaces, ice, sand, and mud) are highly influential in predicting crash outcomes. These indicate that road quality and conditions play a major role in accident contributory causes.
- **Crash Day of the Week**: The day of the week, particularly specific days like **Day 2** and **Day 7**, also emerged as important, potentially revealing temporal patterns related to accident likelihood.
- **Injury Severity**: The feature **MOST_SEVERE_INJURY_REPORTED, NOT EVIDENT** and **injuries_binned_No Injuries** highlight the role of injury reporting in model predictions.

**Next Steps: Exploring XGBoost**
While Random Forests have demonstrated considerable improvements, there is potential for further enhancement with **XGBoost**. XGBoost is a highly efficient gradient boosting algorithm that often delivers state-of-the-art performance in classification tasks. It offers several advantages:
- **Handling Class Imbalance**: XGBoost has built-in options for handling class imbalance, such as tuning the `scale_pos_weight` parameter.
- **Boosting Framework**: The boosting framework allows XGBoost to correct mistakes from previous iterations, potentially improving performance on the underrepresented classes.
- **Regularization**: XGBoost includes regularization techniques to prevent overfitting, which is critical when dealing with large feature sets like ours.

By implementing XGBoost, we aim to further refine our model's performance, particularly in predicting minority classes while maintaining or improving overall accuracy.

import xgboost as xgb

# Convert your data into DMatrix format, which is optimized for XGBoost
dtrain = xgb.DMatrix(X_train_pca, label=y_train)
dtest = xgb.DMatrix(X_test_pca, label=y_test)

# Set parameters for XGBoost
params = {
    'objective': 'multi:softmax',  # Multi-class classification
    'num_class': len(y.unique()),   # Number of classes
    'max_depth': 6,                 # Maximum depth of the tree
    'eta': 0.3,                     # Learning rate
    'subsample': 0.8,               # Subsample ratio
    'colsample_bytree': 0.8,        # Column sample ratio
    'seed': 42                      # Random seed
}

# Train the model
xgb_model = xgb.train(params, dtrain, num_boost_round=100)
# Make predictions on the test set
y_pred_xgb = xgb_model.predict(dtest)

# Evaluate the model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
class_report_xgb = classification_report(y_test, y_pred_xgb)

print(f'XGBoost Accuracy: {accuracy_xgb:.4f}')
print('Classification Report:')
print(class_report_xgb)

# Extract PCA components
pca_components = pca.components_

# Extract feature importance from the XGBoost model fitted on PCA data
xgb_feature_importance = xgb_model.get_score(importance_type='weight')

# Create an array for the importance values corresponding to the PCA components
importance_array = np.array([xgb_feature_importance.get(f'f{i}', 0) for i in range(X_train_pca.shape[1])])  # Adjusted for PCA shape

# Calculate the contributions of the original features by multiplying PCA components with importance
pc_importance = importance_array[:, np.newaxis]  # Convert to column vector

# Calculate the contributions of the original features
original_feature_contributions = np.dot(pca_components.T, pc_importance).flatten()

# Create a DataFrame to store the original features and their contributions
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,  # Original feature names
    'Importance': original_feature_contributions
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the top 10 most important original features
print(feature_importance_df.head(10))

# Plot the feature importance for the original features
plt.figure(figsize=(20, 10))
plt.barh(feature_importance_df['Feature'].head(10), feature_importance_df['Importance'].head(10))
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top 10 Features by Importance')
plt.show()

**Model Comparison: Random Forest vs. XGBoost**

- **Improvement in Class Predictions**: The XGBoost model outperformed Random Forest with an accuracy of **63.93%**. It showed improvements in precision, recall, and F1-scores for several classes, particularly for Pedestrian/Cyclist Errors, where it achieved a precision of **0.67** and recall of **0.82**.
- **Handling Class Imbalance**: XGBoost demonstrated better handling of class imbalance compared to Random Forest, as indicated by improved predictions across classes. It was particularly effective for the classes where Random Forest struggled, such as Driver Error and Environmental Factors.
- **Drawbacks**: Despite its improved performance, XGBoost may also encounter challenges with interpretability due to its complexity, and it could be computationally intensive compared to Random Forest.

**Summary of Feature Importance**

**Key Insights:**
- **Road Conditions**: Features related to road conditions, like **ROAD_DEFECT_WORN SURFACE** and **CRASH_DAY_OF_WEEK**, emerged as critical predictors, reinforcing the importance of road quality in accident outcomes.
- **Temporal Patterns**: The day of the week was a significant predictor, particularly for specific days like **Day 2** and **Day 7**, suggesting that temporal patterns are relevant to accident occurrences.
- **Severity of Injuries**: The feature **MOST_SEVERE_INJURY_REPORTED, NOT EVIDENT** highlighted its importance in predicting outcomes, suggesting that injury reporting can significantly impact model predictions.

**Next Steps: Exploring Neural Networks**
Given the strong performance of XGBoost, the next step involves implementing a neural network model to further enhance predictive capabilities.


from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
import tensorflow as tf

# Check if GPU is available
gpu_available = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpu_available))


# Create a Sequential model
model = keras.Sequential()

# Input layer
model.add(layers.Input(shape=(X_train_pca.shape[1],)))

# Hidden layers
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.2))  # Dropout for regularization
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))


# Output layer
model.add(layers.Dense(len(y.unique()), activation='softmax'))  # softmax for multi-class classification
model.compile(loss='sparse_categorical_crossentropy',  # Use sparse_categorical_crossentropy for given our label encoding
              optimizer=optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])
# Train the model
history = model.fit(X_train_pca, y_train, epochs=50, batch_size=32, validation_split=0.2)
# Make predictions
y_pred_nn = np.argmax(model.predict(X_test_pca), axis=-1)

# Evaluate performance
accuracy_nn = accuracy_score(y_test, y_pred_nn)
conf_matrix_nn = confusion_matrix(y_test, y_pred_nn)
class_report_nn = classification_report(y_test, y_pred_nn)

print(f'Neural Network Accuracy: {accuracy_nn:.4f}')
print('Classification Report:')
print(class_report_nn)
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

**Model Comparison: Neural Network vs. XGBoost**

**Neural Network Performance**

- **Accuracy**: The Neural Network achieved an accuracy of **64.29%**, indicating a slight improvement over the XGBoost model, which had an accuracy of **63.93%**. The training accuracy is steadily increasing and seems to stabilize around 64.0%.The validation accuracy starts lower but does not improve significantly and remains around 63.5% to 64%. There is some gap between training and validation accuracy, which could suggest overfitting if the training accuracy is substantially higher than the validation accuracy.

**Loss**:

The training loss decreases consistently and approaches a lower bound (around 0.74). The validation loss, on the other hand, also decreases but seems to stabilize at a higher value than the training loss, indicating that the model is not generalizing well.
  
**Comparison with XGBoost**

- **XGBoost Performance**: The XGBoost model performed slightly worse in terms of accuracy but provided a more balanced performance across classes and better handling of class imbalance.

Suggested Improvement for Neural Network

- **Implement Callbacks to Prevent Overfitting**:
   - `EarlyStopping`  during training will monitor validation loss and halt training when it starts to increase. This can help prevent overfitting and ensure the model retains the best weights.
   

# Check if GPU is available
gpu_available = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpu_available))

# Create a Sequential model
model = keras.Sequential()

# Input layer
model.add(layers.Input(shape=(X_train_pca.shape[1],)))

# Hidden layers
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.2))  # Dropout for regularization
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))

# Output layer
model.add(layers.Dense(len(y.unique()), activation='softmax'))  # Softmax for multi-class classification

# Compile the model
model.compile(
    loss='sparse_categorical_crossentropy',  # Sparse categorical cross-entropy for label encoding
    optimizer=optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# Define callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=5,           # Stop after 5 epochs with no improvement
    restore_best_weights=True  # Restore the best model weights
)

checkpoint = callbacks.ModelCheckpoint(
    'best_model.keras',  # Save the best model
    monitor='val_loss',
    save_best_only=True,  # Save only when val_loss improves
    mode='min'
)

# Train the model with callbacks
history = model.fit(
    X_train_pca, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, checkpoint]
)

# Make predictions
y_pred_nn = np.argmax(model.predict(X_test_pca), axis=-1)

# Evaluate performance
accuracy_nn = accuracy_score(y_test, y_pred_nn)
class_report_nn = classification_report(y_test, y_pred_nn)

print(f'Neural Network Accuracy: {accuracy_nn:.4f}')
print('Classification Report:')
print(class_report_nn)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

**Summary of Improvements in Neural Network Performance**

**Model Comparison: Updated vs. Previous Neural Network**

| Metric                     | Previous Model | Updated Model | Improvement       |
|----------------------------|----------------|---------------|--------------------|
| **Accuracy**               | 0.6400         | **0.6437**    | Increased by 0.0037 |
| **Precision (Class 1)**    | 0.57           | **0.57**      | Unchanged           |
| **Precision (Class 4)**    | 0.67           | **0.68**      | Increased by 0.01   |
| **Recall (Class 1)**       | 0.45           | **0.45**      | Unchanged           |
| **Recall (Class 4)**       | 0.81           | **0.80**      | Decreased by 0.01   |
| **F1-score (Class 1)**     | 0.50           | **0.51**      | Increased by 0.01   |
| **F1-score (Class 4)**     | 0.73           | **0.73**      | Unchanged           |
| **Macro Average (F1-score)** | 0.29         | 0.29          | Unchanged           |
| **Weighted Average (F1-score)** | 0.62      | 0.62          | Unchanged           |



**Conclusion**

The updated neural network model demonstrated a slight overall improvement in accuracy, increasing from **0.6400** to **0.6437**. Key enhancements were observed in precision and F1-score for Class 4, indicating that the model's performance in identifying this class has improved.

**Overfitting Analysis:**
From the accuracy and loss graphs, we can observe the following:
- **Model Accuracy**: The training accuracy shows a steady increase, while the validation accuracy plateaus around **0.645**. This suggests that the model is performing well on training data but struggles to generalize to unseen data, indicative of potential overfitting.
- **Model Loss**: The training loss decreases consistently, while the validation loss does not show a similar decline and remains relatively constant. This disparity further suggests that the model may be memorizing the training data rather than learning to generalize.

To address these issues:
- **Regularization Techniques**: Implementing techniques such as dropout (already used) and L2 regularization can help combat overfitting.
- **Early Stopping**: Utilizing early stopping could prevent the model from training for too long, halting the training when the validation performance begins to degrade.

Overall, the adjustments have yielded a marginal but meaningful enhancement in predictive capabilities, paving the way for improved safety outcomes through better understanding of contributory factors in traffic accidents. However, careful attention is needed to improve generalization and mitigate overfitting in future iterations.

# 5. Model Validation
# Define the accuracies for each model
model_f1_scores = {
    'Baseline Dummy Classifer': 0.5953,
    'Logistic Regression': 0.6303,
    'Logistic Regression with SMOTE': 0.3065,
    'Random Forest': 0.6140,
    'XGBoost': 0.6393,
    'Neural Network': 0.6400
}


model_performance_df = pd.DataFrame(model_f1_scores.items(), columns=['Model', 'Accuracy'])


plt.figure(figsize=(15, 8))
plt.bar(model_performance_df['Model'], model_performance_df['Accuracy'], color=['blue', 'green', 'orange', 'red'])
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.axhline(y=0.5, color='r', linestyle='--')
plt.show()

**Model Performance Overview**

In this analysis, we evaluated several classification models to determine the best fit for predicting traffic crash contributory causes. Below is a summary of the accuracy and key insights for each model:

| Model                         | Accuracy | Key Insights                                                                                           |
|-------------------------------|----------|--------------------------------------------------------------------------------------------------------|
| **Logistic Regression**       | 0.6303   | Struggled with most classes; good recall for **Pedestrian/Cyclist Errors** but poor overall performance. |
| **Logistic Regression (SMOTE)** | 0.3066   | Significant drop in accuracy; failed to effectively address class imbalance.                           |
| **Random Forest**             | 0.6140   | Good recall for **Pedestrian/Cyclist Errors**; poor performance on minority classes.                   |
| **XGBoost**                   | 0.6393   | Comparable to the neural network; maintains good performance for certain classes, particularly **Pedestrian/Cyclist Errors**. |
| **Neural Network**            | 0.6429   | Highest accuracy; performs well for **Pedestrian/Cyclist Errors** but shows signs of potential overfitting. |

**Conclusion**

1. **Top Performer**: The **Neural Network** achieved the highest accuracy (0.6429) among all models and demonstrated strong performance in classifying key categories. However, it exhibits some signs of overfitting.

2. **Close Contender**: **XGBoost** closely follows with an accuracy of 0.6393, showcasing resilience against overfitting and robust classification capabilities.

3. **Next Best**: **Logistic Regression** (0.6303) displayed reasonable performance but lacked robustness across other classes.

4. **Random Forest** (0.6140) effectively identified **Pedestrian/Cyclist Errors** but struggled with minority classes.

5. **Logistic Regression (SMOTE)** performed poorly (0.3066), indicating that SMOTE did not effectively resolve class imbalance.

**Final Recommendation**

Based on the analysis, the **Neural Network** is recommended as the best model for predicting traffic crash causes, with **XGBoost** as a strong alternative. Future steps should include hyperparameter tuning to enhance model performance and mitigate overfitting, as well as employing model interpretability techniques to better understand decision-making processes.

# 6. Conclusion and Recommendations
## **Conclusion and Recommendations**

### **Conclusion**
1. **Addressing the Problem with Predictive Models**:
   - The project’s objective was to **predict the primary causes of accidents** to help traffic planners and policymakers design targeted interventions. Both the **Neural Network** and **XGBoost models** effectively captured critical accident causes, such as **road conditions, time of day, and human behavior**, aligning with the stakeholders’ need for actionable insights.
   - **Neural Network** achieved the highest accuracy (**0.6429**) by learning complex, non-linear patterns from the data, helping identify nuanced relationships between variables. However, it exhibited **overfitting**, suggesting that further tuning is needed for consistent performance.
   - **XGBoost** followed closely with an accuracy of **0.6393**, providing robust performance without significant overfitting, making it a reliable alternative for practical applications.

2. **Insights on Contributory Causes**:
   - Key features identified by the models, such as **road defects** and **day of the week**, align with real-world safety concerns. This demonstrates that the models are not only predictive but also relevant to stakeholder needs.
   - These insights help city planners and safety boards focus on high-impact areas such as **infrastructure repair** (road defects) and **time-based interventions** (e.g., weekend traffic management).

3. **Handling Data Challenges**:
   - **Class Imbalance**: Despite efforts like **SMOTE**, models such as **Logistic Regression** struggled with minority classes, which reflects the complexity of accurately modeling rare accident causes.
   - The **Neural Network** and **XGBoost** outperformed other models by maintaining reasonable performance across different categories, demonstrating their ability to handle data imbalance better, though further improvement is still needed.

---

### **Recommendations for Future Work**
1. **Hyperparameter Tuning**: Further refine the **Neural Network** to address overfitting and unlock additional performance gains.
2. **Feature Engineering**: Explore new features, such as **weather and traffic congestion interactions**, to capture more nuanced relationships between accident causes.
3. **Continuous Learning**: As new data becomes available, retrain models periodically to maintain predictive relevance and adapt to changing traffic patterns.


