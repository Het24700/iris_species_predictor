import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("🌸 Iris Dataset Analysis")

# Load Dataset
df = pd.read_csv("IRIS.csv")

st.header("Dataset")
st.dataframe(df)

# ---------------- Chart 1 ----------------
st.subheader("Species Count")

fig, ax = plt.subplots()
df["species"].value_counts().plot(kind="bar", ax=ax)
ax.set_xlabel("Species")
ax.set_ylabel("Count")
st.pyplot(fig)

# ---------------- Chart 2 ----------------
st.subheader("Sepal Length Distribution")

fig, ax = plt.subplots()
ax.hist(df["sepal_length"], bins=10)
ax.set_xlabel("Sepal Length")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# ---------------- Chart 3 ----------------
st.subheader("Petal Length Distribution")

fig, ax = plt.subplots()
ax.hist(df["petal_length"], bins=10)
ax.set_xlabel("Petal Length")
ax.set_ylabel("Frequency")
st.pyplot(fig)