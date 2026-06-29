import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score

# 1. Page Configuration (Ensures symmetry across wide or centered screens)
st.set_page_config(page_title="Model Evaluation", layout="centered")

# Load and split dataset
df = pd.read_csv('IRIS.csv')
x = df.drop('species', axis=1)
y = df['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# --- Random Forest Model ---
rf = RandomForestClassifier()
model = rf.fit(x_train, y_train)
pred1 = model.predict(x_test)

accuracy = accuracy_score(y_test, pred1)
precision = precision_score(y_test, pred1, average='macro')

# --- Decision Tree Model ---
dc = DecisionTreeClassifier()
model1 = dc.fit(x_train, y_train) # Reusing splits keeps things identical and clean
pred2 = model1.predict(x_test)

accuracy1 = accuracy_score(y_test, pred2)
precision1 = precision_score(y_test, pred2, average='macro')

# =========================================================================
# UI DESIGN SECTION
# =========================================================================

st.title("🎯 Model Performance Metrics")
st.caption("Comparing Random Forest vs. Decision Tree Classifiers on the Iris Dataset.")
st.write("---")

# 1. Random Forest Section
st.subheader('🌲 Random Forest Classifier Evaluation')

rf_col1, rf_col2 = st.columns(2, gap="medium")

with rf_col1:
    with st.container(border=True):
        # Format metric into a percentage string for cleaner look
        st.metric(label="Accuracy", value=f"{accuracy:.2%}")

with rf_col2:
    with st.container(border=True):
        st.metric(label="Precision (Macro)", value=f"{precision:.2%}")

st.write('') # Clean spacing

# 2. Decision Tree Section
st.subheader('🌿 Decision Tree Classifier Evaluation')

dt_col1, dt_col2 = st.columns(2, gap="medium")

with dt_col1:
    with st.container(border=True):
        st.metric(label="Accuracy", value=f"{accuracy1:.2%}")

with dt_col2:
    with st.container(border=True):
        st.metric(label="Precision (Macro)", value=f"{precision1:.2%}")