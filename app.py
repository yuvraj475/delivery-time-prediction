import streamlit as st 
import pandas as pd 
import joblib 

st.set_page_config(
    page_title="Food Delivery Predictor",
    page_icon="🛵",
    layout="wide"
)
st.title("🛵 Food Delivery Time Predictor")
st.subheader('PREDICTED DELIVERY TIME USING MACHINE LEARNING')

st.sidebar.title('ABOUT')
st.sidebar.write('RANDOM FOREST REGRESSION MODEL')
st.sidebar.markdown('## THIS IS NOT  PREDICT ACTUAL TIME IT JUST GIVE A APPROXIMATED TIME')

with st.container(border=True):
    st.subheader('🛵DELIVERY PARTNER')
    age= st.number_input('🧑‍🦱AGE',min_value=18)
    rating = st.number_input('⭐RATING',min_value=1,max_value=5)
    vehicle_rating = st.number_input('🛵VEHICLE RATING',min_value=0,max_value=3)

df = pd.read_csv(
    r"C:\Users\kushw\Downloads\Food delivery cleaned.csv"
)

# st.write(df.columns.tolist())

with st.container(border=True):
    st.subheader('🛣️ ROUTE AND TRAFFIC')
    distance_km =st.number_input('📍DISTANCE',min_value=1,max_value=50)
    road_traffic_density = st.selectbox('🚦 ROAD DENSITY',df['road_traffic_density'].unique())
    city = st.selectbox('🏙️ CITY',df['city'].unique())


with st.container(border=True):
    st.subheader('📦ORDER DETAILS')
    type_of_order  = st.selectbox('🍔 ORDER  TYPE',df['type_of_order'].unique())
    type_of_vehicle = st.selectbox('🛵 VEHICLE TYPE',df['type_of_vehicle'].unique())
    multiple_deliveries =st.number_input('📦MULTIPLE DELIVERY',min_value=0,max_value = 3)


with st.container(border=True):
    st.subheader('🌦️ENVIRONMENT CONDITIONS')
    weatherconditions = st.selectbox('☁️WEATHER CONDITION',df['weatherconditions'].unique())
    festival = st.selectbox('🎇FESTIVAL',df['festival'].unique())


with st.container(border=True):
    st.subheader('⏰TIME DETAILS')
    order_hrs = st.number_input('🕐ORDER TIME',min_value=0,max_value=23)
    week_day=st.selectbox('📅WEEK DAY',df['week_day'].unique())

    input_data = pd.DataFrame({
        'delivery_person_age':[age],
        'delivery_person_ratings':[rating],
        'weatherconditions':[weatherconditions],
        'road_traffic_density':[road_traffic_density],
        'vehicle_condition':[vehicle_rating],
        'type_of_order':[type_of_order],
        'type_of_vehicle':[type_of_vehicle],
        'multiple_deliveries':[multiple_deliveries],
        'festival':[festival],
        'city':[city],
        'week_day':[week_day],
        'order_hrs':[order_hrs],
        'distance_km':[distance_km]

})

model = joblib.load(
    r"C:\Users\kushw\Music\delivery time\delivery_time_pipeline.pkl"
)    

st.divider()
with st.container(border=True):
    st.subheader('🤖 YOUR MODEL IS READY TO GIVE PREDICTION')
    if st.button('📊 PREDICT'):
        
        prediction = model.predict(input_data)
        st.success(f"⏱️ Estimated Delivery Time: {prediction[0]:.2f} minutes")

    
    

