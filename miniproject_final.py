#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries & Initiate Spark Session

# In[26]:


# import sys
# !{sys.executable} -m pip install plotly


# In[27]:


import os
import findspark
findspark.init()


# In[28]:


from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql import SQLContext

from pyspark.sql.types import * 
from pyspark.sql.functions import *

import pandas as pd
from handyspark import * 

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pyspark.mllib.stat import Statistics
# from pyspark.sql.functions import udf
# import pyspark.sql.functions as F
# from pyspark.sql.functions import col, asc,desc
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler,StandardScaler
from pyspark.ml import Pipeline
from sklearn.metrics import confusion_matrix


spark=SparkSession.builder \
.master ("local[*]")\
.appName("miniproject")\
.getOrCreate()
sc=spark.sparkContext
sqlContext=SQLContext(sc)


# In[29]:


bankdeposit_df = spark.read \
 .option("header","True")\
 .option("inferSchema","True")\
 .option("sep",";")\
 .csv("D:\\Libraries\\Documents\\BAN 5753\\Week 12\\XYZ_Bank_Deposit_Data_Classification.csv")
print("There are",bankdeposit_df.count(),"rows",len(bankdeposit_df.columns),
      "columns" ,"in the data.") 


# In[30]:


bankdeposit_df.toPandas().head(10)


# ## Rename Columns That Contain "."

# In[31]:


bankdeposit_df = bankdeposit_df.withColumnRenamed("emp.var.rate","emp_var_rate") \
    .withColumnRenamed("cons.price.idx","cons_price_idx") \
    .withColumnRenamed("cons.conf.idx","cons_conf_idx") \
    .withColumnRenamed("nr.employed","nr_employed")
bankdeposit_df.printSchema()


# ## Encode Education To Ordinal Column

# In[32]:


def udf_multiple(education):
      if (education == 'illiterate'):
        return 1
      elif (education == 'basic.4y'):
        return 2
      elif (education == 'basic.6y'):
        return 3
      elif (education == 'basic.9y'):
        return 4
      elif (education == 'high.school'):
        return 5
      elif (education == 'professional.course'):
        return 6
      elif (education == 'university.degree'):
        return 7
      else: return 0

education_group = udf(udf_multiple)
bankdeposit_df=bankdeposit_df.withColumn("education_group", education_group('education'))
bankdeposit_df=bankdeposit_df.withColumn("education_group",bankdeposit_df.education_group.cast('int'))


# ## Identify Categorical & Numerical Columns

# In[33]:


String_columnList = [item[0] for item in bankdeposit_df.dtypes if item[1].startswith('string')]
Int_columnList = [item[0] for item in bankdeposit_df.dtypes if item[1].startswith('int')]
Double_columnList = [item[0] for item in bankdeposit_df.dtypes if item[1].startswith('double')]

Numerical = Int_columnList + Double_columnList
print("Numerical Columns:",Numerical)
print("String Columns:",String_columnList)


# In[34]:


numeric_features = [t[0] for t in bankdeposit_df.dtypes if t[1] in ('int', 'double')]
bankdeposit_df.select(numeric_features).describe().toPandas().transpose()


# ## Check For Null Values

# In[35]:


null_df = bankdeposit_df.select([count(when(col(c).contains('None') | \
                            col(c).contains('NULL') | \
                            (col(c) == '') | \
                            col(c).isNull() | \
                            isnan(c), c 
                           )).alias(c)
                    for c in bankdeposit_df.columns])

null_df.toPandas().head(1)


# ## Check For Class Imbalance

# In[36]:


bankdeposit_df.groupby('y').count().toPandas()


# ## Note: Many Visualizations Require Conversion To Pandas DF

# In[37]:


bank_df = bankdeposit_df.toPandas()
sns.countplot(data=bank_df, x='y')
plt.show()


# ## Check For Correlation Between Predictors

# In[38]:


numeric_data = bankdeposit_df.select(numeric_features).toPandas()
axs = pd.plotting.scatter_matrix(numeric_data, figsize=(8, 8));
n = len(numeric_data.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())


# ## Univariate Analysis

# In[39]:


numbers = bank_df.select_dtypes(['int64', 'float64']).columns.to_list()
bank_df.hist(figsize=(20,10))
plt.show()

display(bank_df[numbers].describe())


# In[40]:


fig, ax = plt.subplots(3,4, figsize=(20,17))

cat = bank_df.select_dtypes('object').columns.to_list()
cat = cat[:-1]

ax = ax.ravel()
position = 0

for i in cat:
    
    order = bank_df[i].value_counts().index
    sns.countplot(data=bank_df, x=i, ax=ax[position], order=order)
    ax[position].tick_params(labelrotation=90)
    ax[position].set_title(i, fontdict={'fontsize':17})
    
    position += 1

plt.subplots_adjust(hspace=0.7)

plt.show()


# In[41]:


plt.subplot(231)
sns.distplot(bank_df['emp_var_rate'])
fig = plt.gcf()
fig.set_size_inches(20,13)

plt.subplot(232)
sns.distplot(bank_df['cons_price_idx'])
fig = plt.gcf()
fig.set_size_inches(20,13)

plt.subplot(233)
sns.distplot(bank_df['cons_conf_idx'])
fig = plt.gcf()
fig.set_size_inches(20,13)

plt.subplot(234)
sns.distplot(bank_df['euribor3m'])
fig = plt.gcf()
fig.set_size_inches(20,13)

plt.subplot(235)
sns.distplot(bank_df['nr_employed'])
fig = plt.gcf()
fig.set_size_inches(20,13)


# ## Bivariate Analysis 

# In[42]:


import plotly.express as px

fig = px.box(bank_df, x="job", y="duration", color="y")
fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
fig.show()


# In[43]:


fig = px.scatter(bank_df, x="campaign", y="duration", color="y")
fig.show()


# In[44]:


plt.bar(bank_df['month'], bank_df['campaign'])


# In[45]:


sns.violinplot( y=bank_df["marital"], x=bank_df["cons_price_idx"])


# In[46]:


bank_yes = bank_df[bank_df['y']=='yes']

df1 = pd.crosstab(index = bank_yes["marital"],columns="count")    
df2 = pd.crosstab(index = bank_yes["month"],columns="count")  
df3= pd.crosstab(index = bank_yes["job"],columns="count") 
df4=pd.crosstab(index = bank_yes["education"],columns="count")

fig, axes = plt.subplots(nrows=2, ncols=2)
df1.plot.bar(ax=axes[0,0])
df2.plot.bar(ax=axes[0,1])
df3.plot.bar(ax=axes[1,0])
df4.plot.bar(ax=axes[1,1]) 
fig.set_size_inches(15,10)


# ## Prepare Inputs For Model

# In[47]:


from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
categoricalColumns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome', 'month', 'day_of_week']
stages = []

for categoricalCol in categoricalColumns:    
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]


# In[48]:


label_stringIdx = StringIndexer(inputCol = 'y', outputCol = 'label')
stages += [label_stringIdx]
numericCols = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed']
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]


# In[49]:


cols = bankdeposit_df.columns
from pyspark.ml import Pipeline
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(bankdeposit_df)
bankdeposit_df = pipelineModel.transform(bankdeposit_df)
selectedCols = ['label', 'features'] + cols
bankdeposit_df = bankdeposit_df.select(selectedCols)
bankdeposit_df.printSchema()


# In[50]:


from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(inputCol = 'features', outputCol = 'scaledFeatures')
scaler_model = scaler.fit(bankdeposit_df)
bankdeposit_df = scaler_model.transform(bankdeposit_df)
bankdeposit_df.toPandas().head(3)


# In[51]:


pd.DataFrame(bankdeposit_df.take(5), columns=bankdeposit_df.columns).transpose()


# ## Train - Test Split

# In[52]:


train, test = bankdeposit_df.randomSplit([0.7, 0.3], seed = 2018)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))


# ## K-Means Clustering

# In[53]:


from pyspark.ml.clustering import KMeans
kmeans = KMeans(featuresCol = 'scaledFeatures', k=2)
model = kmeans.fit(train)


# In[54]:


from pyspark.ml.evaluation import ClusteringEvaluator

# Make predictions
predictions = model.transform(test)

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))


# In[55]:


centers = model.clusterCenters()
print(centers)


# In[56]:


model.transform(test).select('scaledFeatures', 'prediction').show()


# ## Logistic Regression

# In[57]:


from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
lrModel = lr.fit(train)


# In[58]:


import matplotlib.pyplot as plt
import numpy as np
beta = np.sort(lrModel.coefficients)
plt.plot(beta)
plt.ylabel('Beta Coefficients')
plt.show()


# In[59]:


trainingSummary = lrModel.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))


# In[60]:


pr = trainingSummary.pr.toPandas()
plt.plot(pr['recall'],pr['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()


# In[61]:


predictions = lrModel.transform(test)
predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)


# In[62]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
print('Logistic Regression Area Under ROC', evaluator.evaluate(predictions))


# ## Decision Tree

# In[63]:


from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 3)
dtModel = dt.fit(train)
predictions = dtModel.transform(test)
predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)


# In[64]:


evaluator = BinaryClassificationEvaluator()
print("Decision Tree Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))


# ## Random Forest

# In[65]:


from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
rfModel = rf.fit(train)
predictions = rfModel.transform(test)
predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)


# In[66]:


evaluator = BinaryClassificationEvaluator()
print("Random Forest Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))


# ## Gradient-Boosted Tree

# In[67]:


from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(maxIter=10)
gbtModel = gbt.fit(train)
predictions = gbtModel.transform(test)
predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)


# In[68]:


evaluator = BinaryClassificationEvaluator()
print("Gradient-Boosted Tree Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))


# In[69]:


from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
paramGrid = (ParamGridBuilder()
             .addGrid(gbt.maxDepth, [2, 4, 6])
             .addGrid(gbt.maxBins, [20, 60])
             .addGrid(gbt.maxIter, [10, 20])
             .build())
cv = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
# Run cross validations.  This can take about 6 minutes since it is training over 20 trees!
cvModel = cv.fit(train)
predictions = cvModel.transform(test)
evaluator.evaluate(predictions)


# ## Linear SVM

# In[70]:


from pyspark.ml.classification import LinearSVC
lsvc = LinearSVC(labelCol="label", maxIter=10)
lsvc = lsvc.fit(train)


# In[71]:


pred = lsvc.transform(test)
pred.select('features','label', 'rawPrediction', 'prediction').show(10)


# In[72]:


print("Linear SVM Area Under ROC: " + str(evaluator.evaluate(pred, {evaluator.metricName: "areaUnderROC"})))


# In[73]:


# lr = pipeline.fit(df) // Trained model
# lr.save("/path")
# pipelineModel = lr.load("/path")
# df = pipelineModel.transform(df)

# gbtModel = gbt.fit(train)


# In[90]:


import joblib
# fileName = "gbtModel.joblib"
# joblib.dump(gbtModel, "./" + fileName)


# In[76]:


path = "D:/Libraries/Documents"


# In[89]:


# gbtModel.save("D:/Libraries/Documents")
# gbtModel.save(sc,"D:/Libraries/Documents")


# In[ ]:


sc


# In[88]:


# gbtModel.write().save("D:/Libraries/Documents")


# In[82]:


import pickle


# In[91]:


# pickle.dump(gbtModel, open('D:/Libraries/Documents/gbtModel.pkl', 'wb'))

