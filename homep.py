import streamlit as st 

# 1. Setup Page Layout to Wide Mode
st.set_page_config(page_title="Iris Predictor", layout="wide")

# 2. Sidebar Navigation Panel
with st.sidebar:
    st.title('🎨 Additional Features')
    st.info("Select a module below to explore the dataset and model performance.")
    
    # Clean, stacked native buttons
    if st.button("🌼 Iris Species", use_container_width=True):
        st.switch_page('pages/students1.py')

    if st.button("🎯 Accuracy Score", use_container_width=True):
        st.switch_page('pages/accuracy_score.py')

    if st.button('🧠 Prediction', use_container_width=True):
        st.switch_page('pages/predictions.py')  
        
    if st.button('📈 Visualization', use_container_width=True):
        st.switch_page('pages/charts.py')

# 3. Main Header Section
st.title('🌸 Iris Species Predictor')
st.caption("Leveraging Machine Learning to classify botanical samples with precision.")

# Banner Image (Scales gracefully to the width of the page container)
st.image('https://i.ytimg.com/vi/2HfJVdj3SpM/maxresdefault.jpg', use_container_width=True)

st.write("---")

# 4. Perfectly Symmetrical 50/50 Layout
left_page, right_page = st.columns(2, gap="large")

with left_page:
    st.subheader('📊 What this app does')
    # Native information box providing structured layout
    st.info("💡 **Project Overview**")
    st.markdown("""
    * 🌸 **Predict Species:** Instantly classify Iris flower species.
    * 📊 **Metric Insight:** Real-time model accuracy and evaluation reporting.
    * 🤖 **Algorithm Variety:** Comparison across multiple ML architectures.
    * 📈 **Interactive Design:** Beautiful, user-friendly data discovery.
    * ⚡ **High Performance:** Optimized for real-time inference.
    """)

with right_page:
    st.subheader("🤖 Key Features Section")
    
    # Symmetrical 2x2 grid created with native columns and bordered containers
    row1_col1, row1_col2 = st.columns(2, gap="small")
    with row1_col1:
        with st.container(border=True):
            st.markdown("### 🧠")
            st.markdown("**AI Engine**")
            st.caption("Predictive power")
            
    with row1_col2:
        with st.container(border=True):
            st.markdown("### 📈")
            st.markdown("**Accuracy**")
            st.caption("Model metrics")
            
    row2_col1, row2_col2 = st.columns(2, gap="small")
    with row2_col1:
        with st.container(border=True):
            st.markdown("### 📊")
            st.markdown("**Visuals**")
            st.caption("Data charts")
            
    with row2_col2:
        with st.container(border=True):
            st.markdown("### 📂")
            st.markdown("**Explorer**")
            st.caption("Dataset views")

# 5. Footer
st.write("---")
st.caption("© 2026 AI IRIS Species Dashboard. All Rights Reserved.")