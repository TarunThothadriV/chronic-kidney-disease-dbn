import tensorflow
from tensorflow.keras.layers import Layer
import numpy as np
import pandas as pd
import streamlit as st 
from sklearn import preprocessing
import joblib as jbl
import os
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class RBMLayer(Layer):
    def __init__(self, units, **kwargs):
        super(RBMLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)
        super(RBMLayer, self).build(input_shape)

    def call(self, inputs):
        return tensorflow.nn.sigmoid(tensorflow.matmul(inputs, self.W) + self.b)

    def get_config(self):
        config = super(RBMLayer, self).get_config()
        config.update({'units': self.units})
        return config

#RBM Pre-Training
rbm1 = RBMLayer(units=24,name = 'rbm1')
rbm2 = RBMLayer(units=13,name = 'rbm2')
rbm3 = RBMLayer(units=28,name = 'rbm3')
rbm4 = RBMLayer(units=8,name = 'rbm4')
rbm5 = RBMLayer(units=4,name = 'rbm5')

logistic = layers.Dense(units=2, activation='softmax',name = 'logistic')

model = Sequential([
    layers.Input(shape = (24,)),  # Input layer
    layers.BatchNormalization(),            # Batch normalization
    layers.Dropout(0.2),                    # Dropout layer
    rbm1,                                   # Stacked RBM layer - 1
    rbm2,                                   # Stacked RBM layer - 2
    rbm3,                                   # Stacked RBM layer - 3
    rbm4,                                   # Stacked RBM layer - 4
    rbm5,                                   # Stacked RBM layer - 5
    logistic                                # Final classification layer
])


# Loop through each layer in the model and load the weight
for i, layer in enumerate(model.layers):
    weights = []  # List to store weights for current layer
    j = 0  # Initialize weight index
    # Check if all weight files exist for current layer
    while all(os.path.exists(f'layer_{i}_weights_{j}_{suffix}.npy') for suffix in ['gamma', 'beta', 'moving_mean', 'moving_variance']):
        # Load all weights for current layer
        weight_gamma = np.load(f'layer_{i}_weights_{j}_gamma.npy', allow_pickle=True)
        weight_beta = np.load(f'layer_{i}_weights_{j}_beta.npy', allow_pickle=True)
        weight_mean = np.load(f'layer_{i}_weights_{j}_moving_mean.npy', allow_pickle=True)
        weight_variance = np.load(f'layer_{i}_weights_{j}_moving_variance.npy', allow_pickle=True)
        # Append loaded weights to the list
        weights.append([weight_gamma, weight_beta, weight_mean, weight_variance])
        # Move to next weight index
        j += 1
    # If any weights were loaded for the current layer
    if weights:
        layer.set_weights(weights)



#model = tensorflow.keras.models.load_model('dbn_m.keras')

sscaler = jbl.load('scaler.pkl')
le = jbl.load('label_encoder.pkl')
numerical = ['age','bp','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
nominal = ['sg','al','su','rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']
  
def main(): 
    st.title("Chronic Kidney Disease Predictor")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Chronic Kidney Disease Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)

    age = st.text_input("Age","0")
    bp = st.text_input("Blood Pressure (mm/Hg)","0")
    bgr = st.text_input("Blood Glucose Rate","0")
    bu = st.text_input("Blood Urea","0")
    sc = st.text_input("Serum Creatinine","0")
    sod = st.text_input("Sodium","0")
    pot = st.text_input("Pottasium","0")
    hemo = st.text_input("Hemoglobin","0")
    pcv = st.text_input("Packed Cells Volume","0")
    wc = st.text_input("WBC Count","0")
    rc = st.text_input("RBC Count","0")

    sg = st.selectbox("Specific Gravity",["1.005","1.010","1.015","1.020","1.025"])
    al = st.selectbox("Albumin",["0","1","2","3","4","5"])
    su = st.selectbox("Sugar",["0","1","2","3","4","5"])
    rbc = st.selectbox("Red Blood Cells",["normal","abnormal"])
    pc = st.selectbox("Pus Cells",["normal","abnormal"])
    pcc = st.selectbox("Pus Cells Clumps",["present","notpresent"])
    ba = st.selectbox("Bacteria",["present","notpresent"])
    htn = st.selectbox("HyperTension",["yes","no"])
    dm = st.selectbox("Diabetes Mellitus",["yes","no"])
    cad = st.selectbox("Coronary Artery Disease",["yes","no"])
    appet = st.selectbox("Appetite",["good","poor"])
    pe = st.selectbox("Pedal Edema",["yes","no"])
    ane = st.selectbox("Anemia",["yes","no"])
        
    if st.button("Predict"): 
        features = [[age,bp,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc,sg,al,su,rbc,pc,pcc,ba,htn,dm,cad,appet,pe,ane,]]
        data = {'age': int(age),
                'bp': bp,
                'bgr':int(bgr),
                'bu':int(bu),
                'sc':float(sc),
                'sod':int(sod),
                'pot':float(pot),
                'hemo':float(hemo),
                'pcv':int(pcv),
                'wc':int(wc),
                'rc':float(rc),
                'sg':sg,
                'al':int(al),
                'su':su,
                'rbc':rbc,
                'pc':pc,
                'pcc':pcc,
                'ba':ba,
                'htn':htn,
                'dm':dm,
                'cad':cad,
                'appet':appet,
                'pe':pe,
                'ane':ane
                }
        #print(data)
        df=pd.DataFrame([list(data.values())], columns=['age','bp','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc','sg','al','su','rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane'])
        for nom in nominal:
            df[nom] =df[nom].replace(to_replace = {'yes' : 1, 'no' : 0,'normal':1,'abnormal':0,'present':1,'notpresent':0,'good':0,'poor':1,'1.005':0,'1.010':1,'1.015':2,'1.020':3,'1.025':4})

        # Assuming df is your DataFrame with one row
        row_values = df.iloc[0].values  # Get the values of the first row
        row_values_1d = row_values.reshape(-1)  # Reshape the values to a 1D array if needed

        # Now you can use the LabelEncoder to transform the 1D array
        #transformed_values = le.transform(row_values_1d)

        features = sscaler.transform(df.values)
        #features_list = df.values.tolist()      
        prediction = model.predict(features)

        output = prediction[0]
        if output[1] > output[0]:
            text = "You're Safe! No Chronic Kidney Disease"
        else:
            text = "Consult your doctor as you have high chance of Chronic Kidney Disease !!"
        st.success('{}'.format(text))
      
if __name__=='__main__': 
    main()
