import streamlit as st
st.title('Iris species predictor')
sepl = st.number_input("Sepal Length")
sepw = st.number_input("Sepal Width")
petl = st.number_input("Petal Length")
petw = st.number_input("Petal Width")
pred=st.button('Predict')
if pred:
    import streamlit as st
    import pandas as pd
    df=pd.read_csv('IRIS.csv')
    from sklearn.model_selection import train_test_split
    x=df.drop('species',axis=1)
    y=df['species']
    from sklearn.ensemble import RandomForestClassifier
    rf=RandomForestClassifier()
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    model=rf.fit(x_train,y_train)
    pred1=model.predict([[sepl,sepw,petw,petl]])
    from sklearn.tree import DecisionTreeClassifier
    dc= DecisionTreeClassifier()
    x1_train,x1_test,y1_train,y1_test=train_test_split(x,y,test_size=0.2,random_state=42)
    model1=dc.fit(x1_train,y1_train)
    if sepl and sepw and petl and petw:
        pred = model.predict([[float(sepl), float(sepw), float(petl), float(petw)]])
        st.write(f'Predicted speicies:{pred}')

