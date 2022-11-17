## Mini-Project-2
OKState BAN 5753 Mini-Project 2: Pyspark MLlib



![1](Hawks/1.jpg)










We begin by importing the necessary libraries, initiating a Spark session, and reading in the data. Here, we can see what the data looks like: 
![1](Images/1.PNG)

The first data preprocessing step is to rename all columns that contain a “.”, because this character will cause problems later. We will replace it with an underscore: 
![2](Images/2.PNG)

Next, we will encode the Education column into a new integer column. Since levels of education have an implied order, we can provide additional meaning to this column by encoding it into an ordinal variable: 
![3](Images/3.PNG)
After doing this, we will look at the various columns, based on their data type. First, we will look at the categorical/string columns: 
![4](Images/4.PNG)
We also will examine the numerical columns and their statistics: 
![5](Images/5.PNG)
 

After this step, we check for null or missing values in any columns. We can see that there are no null values: 
![6](Images/6.PNG)
We also can check for imbalance between the two classes in the outcome variable, “y”. We see some class imbalance, so we can adjust for this with oversampling if it becomes and issue when we are training and evaluating our models.  
![7](Images/7.PNG)
We also can visualize this class imbalance. For many of the visualizations in this report, we will need to convert the PySpark dataframe to Pandas in order to use the Pandas visualization functions, as seen below: 
![8](Images/8.PNG)
The final step for us to check before we begin more detailed analysis is for correlation between predictor variables. If several predictors are highly correlated, then we can exclude some of them from our analysis without loosing explanation power. Below, our correlation analysis can be seen: 
![9](Images/9.PNG)

![10](Images/10.PNG)
![11](Images/11.PNG)
![12](Images/12.PNG)
![13](Images/13.PNG)
![14](Images/14.PNG)
![15](Images/15.PNG)
![16](Images/16.PNG)
![17](Images/17.PNG)
![18](Images/18.PNG)
![19](Images/19.PNG)
![20](Images/20.PNG)
![21](Images/21.PNG)

![22](Images/22.PNG)

![23](Images/23.PNG)

![24](Images/24.PNG)

![25](Images/25.PNG)
