# %%
import pandas as pd
import numpy as np
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')





# %%
df =pd.read_csv("loan.csv")
df

# %%
df.head()

# %%
df.tail()

# %%
df.shape

# %%
df.columns

# %%
df.info()

# %%
df.isnull().sum()

# %%
#check outlier
plt.figure(figsize=(12,8))
sns.boxplot(data=df)
plt.show()


# %%
#fill null value
df['LoanAmount']=df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History']=df['Credit_History'].fillna(df['Credit_History'].mean())

# %%
df.isnull().sum()

# %%
#fill null of object dtype
df['Gender']=df['Gender'].fillna(df['Gender'].mode()[0])
df['Married']=df['Married'].fillna(df['Married'].mode()[0])
df['Dependents']=df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed']=df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

# %%
df.isnull().sum()

# %%
print('number of people who took loan by gender')
print(df['Gender'].value_counts())
sns.countplot(x='Gender',data=df)
plt.show()

# %%
print('number of people who took loan by ')
print(df['Married'].value_counts())
sns.countplot(x='Married',data=df)
plt.show()

# %%
# Keep only numeric columns
corr = df.select_dtypes(include=['number']).corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True)
plt.show()

# %%
df['Total_Income']=df['ApplicantIncome'] + df['CoapplicantIncome']
df.head()

# %%
#apply log transformation
df['ApplicantIncomelog']=np.log(df['ApplicantIncome']+1)
sns.distplot(df['ApplicantIncomelog'])
plt.show()

# %%
df['LoanAmountlog']=np.log(df['LoanAmount']+1)
sns.distplot(df['LoanAmountlog'])
plt.show()

# %%
df['Loan_Amount_Termlog']=np.log(df['Loan_Amount_Term']+1)
sns.distplot(df['Loan_Amount_Termlog'])
plt.show()

# %%
df['Total_Income_log']=np.log(df['Total_Income']+1)
sns.distplot(df['Total_Income_log'])
plt.show()

# %%
#drop unecessary columns
cols=['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Total_Income','Loan_ID']
df=df.drop(columns=cols,axis=1)
df.head()

# %%
#encoding technique:label encoding,one hot encoding
from sklearn.preprocessing import LabelEncoder
cols=['Gender','Married','Education','Dependents','Self_Employed','Property_Area','Loan_Status']
le=LabelEncoder()
for col in cols:
    df[col]=le.fit_transform(df[col])

# %%
df.head(5)

# %%
df.dtypes

# %%
#split Independent and dependent feature

x=df.drop(columns=['Loan_Status'],axis=1)
y=df['Loan_Status']


# %%
x

# %%
y

# %%
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# %%
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

# %%
#logistic regression
model1=LogisticRegression()
model1.fit(x_train,y_train)
y_pred_model1=model1.predict(x_test)
accuracy=accuracy_score(y_test,y_pred_model1)

# %%
accuracy*100

# %%
#Accuracy:ratio of correctly predicted values to the total values

# %%
score=cross_val_score(model1,x,y,cv=5)
score

# %%
np.mean(score)*100

# %%
#decision tree classifier
model2=DecisionTreeClassifier()
model2.fit(x_train,y_train)
y_pred_model2=model2.predict(x_test)
accuracy=accuracy_score(y_pred_model2,y_test)
print("accuracy score of decision tree model:",accuracy*100)

# %%
score=cross_val_score(model2,x,y,cv=5)
print("cross validation of decison tree:",np.mean(score)*100)

# %%
#random forest classifier
model3=RandomForestClassifier()
model3.fit(x_train,y_train)
y_pred_model3=model3.predict(x_test)
accuracy=accuracy_score(y_pred_model3,y_test)
print("accuracy score of random forest model:",accuracy*100)

# %%
#KNeighbor model
model4=KNeighborsClassifier(n_neighbors=3)
model4.fit(x_train,y_train)
y_pred_model4=model4.predict(x_test)
accuracy=accuracy_score(y_pred_model4,y_test)
print("accuracy score of KNeighbors model:",accuracy*100)

# %%
score=cross_val_score(model4,x,y,cv=5)
print("cross validation of KNNeighbor :",np.mean(score)*100)

# %%
from sklearn.metrics import classification_report

def generate_classification_report(model_name,y_test,y_pred):
    report = classification_report(y_test,y_pred)
    print(f"Classfication Report For {model_name}:\n{report}\n")

generate_classification_report(model1,y_test,y_pred_model1)
generate_classification_report(model2,y_test,y_pred_model2)
generate_classification_report(model3,y_test,y_pred_model3)
generate_classification_report(model4,y_test,y_pred_model4)

# %%
df['Loan_Status'].value_counts()

# %%
from imblearn.over_sampling import RandomOverSampler

# %%
oversample = RandomOverSampler(random_state=42)
x_resampled,y_resampled=oversample.fit_resample(x,y)

df_resampled=pd.concat([pd.DataFrame(x_resampled,columns=x.columns),pd.Series(y_resampled,name="Loan_Status")],axis=1)

# %%
x_resampled

# %%
y_resampled

# %%
y_resampled.value_counts()

# %%
x_resampled_train,x_resampled_test,y_resampled_train,y_resampled_test=train_test_split(x_resampled,y_resampled,test_size=0.25,random_state=42)

# %%
#logistic regression
model1=LogisticRegression()
model1.fit(x_resampled_train,y_resampled_train)
y_pred_model1=model1.predict(x_resampled_test)
accuracy=accuracy_score(y_resampled_test,y_pred_model1)
accuracy*100

# %%
#decision tree classifier
model2=DecisionTreeClassifier()
model2.fit(x_resampled_train,y_resampled_train)
y_pred_model2=model2.predict(x_resampled_test)
accuracy=accuracy_score(y_pred_model2,y_resampled_test)
print("accuracy score of decision tree model:",accuracy*100)

# %%
#random forest classifier
model3=RandomForestClassifier()
model3.fit(x_resampled_train,y_resampled_train)
y_pred_model3=model3.predict(x_resampled_test)
accuracy=accuracy_score(y_pred_model3,y_resampled_test)
print("accuracy score of random forest model:",accuracy*100)

# %%
#KNeighbor model
model4=KNeighborsClassifier(n_neighbors=3)
model4.fit(x_resampled_train,y_resampled_train)
y_pred_model4=model4.predict(x_resampled_test)
accuracy=accuracy_score(y_pred_model4,y_resampled_test)
print("accuracy score of KNeighbors model:",accuracy*100)

# %%
from sklearn.metrics import classification_report

def generate_classification_report(model_name,y_resampled_test,y_pred):
    report = classification_report(y_resampled_test,y_pred)
    print(f"Classfication Report For {model_name}:\n{report}\n")

generate_classification_report(model1,y_resampled_test,y_pred_model1)
generate_classification_report(model2,y_resampled_test,y_pred_model2)
generate_classification_report(model3,y_resampled_test,y_pred_model3)
generate_classification_report(model4,y_resampled_test,y_pred_model4)







import streamlit as st
import numpy as np
import pandas as pd
import joblib

@st.cache_resource
def load_model():
   # Train your model (e.g., RandomForestClassifier)
   from sklearn.ensemble import RandomForestClassifier
   model = RandomForestClassifier()
   model.fit(x_train, y_train)

modelx = load_model()

st.title("Loan Prediction App")

def preprocess_input(df):
    # ...same preprocessing as before...
    df_processed = df.copy()
    df_processed['Gender'] = df_processed['Gender'].map({'Male': 1, 'Female': 0})
    df_processed['Married'] = df_processed['Married'].map({'Yes': 1, 'No': 0})
    df_processed['Dependents'] = df_processed['Dependents'].replace('3+', 3).astype(int)
    df_processed['Education'] = df_processed['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    df_processed['Self_Employed'] = df_processed['Self_Employed'].map({'Yes': 1, 'No': 0})
    df_processed['Property_Area'] = df_processed['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
    df_processed['TotalIncome'] = df_processed['ApplicantIncome'] + df_processed['CoapplicantIncome']
    df_processed['Log_ApplicantIncome'] = np.log(df_processed['ApplicantIncome'] + 1)
    df_processed['Log_LoanAmount'] = np.log(df_processed['LoanAmount'] + 1)
    df_processed['Log_Loan_Amount_Term'] = np.log(df_processed['Loan_Amount_Term'] + 1)
    df_processed['Log_TotalIncome'] = np.log(df_processed['TotalIncome'] + 1)

    features = df_processed[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                             'Credit_History', 'Property_Area', 'Log_ApplicantIncome',
                             'Log_LoanAmount', 'Log_Loan_Amount_Term', 'Log_TotalIncome']]
    return features

# File uploader with button trigger
uploaded_file = st.file_uploader("Upload loan.csv", type=["csv"])

if uploaded_file is not None:
    if st.button("Predict from CSV"):
        df = pd.read_csv(uploaded_file)
        try:
            input_df = preprocess_input(df)
            predictions = modelx.predict(input_df)
            df["Prediction"] = predictions
            st.subheader("Predictions")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error during prediction: {e}")

st.markdown("---")
st.subheader("Or enter details manually")

# Collect manual inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Amount Term (in months)", min_value=0)

if st.button("Predict Loan Approval"):
    total_income = applicant_income + coapplicant_income
    features = [
        1 if gender == "Male" else 0,
        1 if married == "Yes" else 0,
        {"0": 0, "1": 1, "2": 2, "3+": 3}[dependents],
        1 if education == "Graduate" else 0,
        1 if self_employed == "Yes" else 0,
        float(credit_history),
        {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area],
        np.log(applicant_income + 1),
        np.log(loan_amount + 1),
        np.log(loan_term + 1),
        np.log(total_income + 1)
    ]
    prediction = modelx.predict([features])[0]
    st.success("Loan Approved ✅" if prediction == 1 else "Loan Rejected ❌")
