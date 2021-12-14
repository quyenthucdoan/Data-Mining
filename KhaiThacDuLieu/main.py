import streamlit as st
import pandas as pd
import Preprocessing as pre
import pickle
import Visualization as vs
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import tree as tree1
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

import kohonen as koh

# Bien toan cuc

html_heading = '''
# Khai thác dữ liệu
## Phân tích dữ liệu về sự hài lòng của hành khách đi máy bay dựa trên đánh giá về 15 tiêu chí.
### Gồm có các chức năng chính:
* Trực quan hóa dữ liệu với biểu đồ Countplot và Catplot.
* Kết quả các độ đo(Precision, Recall, F1-score) giữa 3 thuật toán máy học(Logistic Regression, Decision Tree, Random Forest).
* Trực quan hóa các thuộc tính quan trọng nhất ảnh hưởng đến nhãn phân lớp của bài toán(Important Features).
* Dự đoán được giá trị y dựa vào các thuật toán như: Logistic Regression, Decision Tree, Random Forest.
* Thực hiện phân cụm bằng hai thuật toán Kmeans và Kohonen
* So sánh độ dự đoán chính xác giữa Naive Bayes và Decision Tree
'''
st.markdown(html_heading,unsafe_allow_html=True)
st.markdown('<style>h1{color: #196594;}</style>', unsafe_allow_html=True)
st.markdown('<style>h2{color: #EDB200;}</style>', unsafe_allow_html=True)
st.markdown('<style>h3{color: #D01013;}</style>', unsafe_allow_html=True)
st.markdown('<style>li{color: black;}</style>', unsafe_allow_html=True)
st.markdown('<style>h4{color: #CD5C5C;}</style>', unsafe_allow_html=True)
#choose data
def user_input_features():
    # Gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    CustomerType = st.sidebar.selectbox('Customer Type', ('Loyal Customer', 'disloyal Customer'))
    Age = st.sidebar.slider('Age', 7, 85)
    TypeofTravel = st.sidebar.selectbox('Type of Travel', ('Personal Travel', 'Business travel'))
    Class = st.sidebar.selectbox('Class', ('Eco Plus', 'Business','Eco'))
    FlightDistance = st.sidebar.slider('FlightDistance', 31, 4983)
    # Inflight_wifi_service = st.sidebar.slider('Inflight wifi service', 1,5)
    # Inflight_wifi_service = 1;
    # Departure_Arrival_time_convenient = st.sidebar.slider('Departure/Arrival time convenient',1,5)
    # Gatelocation = st.sidebar.slider('Gate location', 1,5)
    # Ease_of_Online_booking = st.sidebar.slider('Ease of Online booking', 1,5)
    Food_and_drink = st.sidebar.slider('Food and drink', 1,5)
    Onlineboarding = st.sidebar.slider('Online boarding', 1,5)
    Seat_comfort = st.sidebar.slider('Seat comfort', 1,5)
    Inflight_entertainment = st.sidebar.slider('Inflight entertainment', 1,5)
    On_board_service = st.sidebar.slider('On-board service', 1,5)
    Leg_room_service = st.sidebar.slider('Leg room service', 1,5)
    Baggage_handling = st.sidebar.slider('Baggage handling', 1,5)
    Checkin_service = st.sidebar.slider('Checkin service', 1,5)
    Inflight_service = st.sidebar.slider('Inflight service', 1,5)
    Cleanliness = st.sidebar.slider('Cleanliness', 1,5)
    # Arrival_Delay_in_Minutes = st.sidebar.slider('Arrival Delay in Minutes', 6, 18, 23)
    # Departure_Delay_in_Minutes = st.sidebar.slider('Departure Delay in Minutes', 0, 109)
    datacl = {'Gender': 'Male',
              'Customer Type': CustomerType,
              'Age': Age,
              'Type of Travel': TypeofTravel,
              'Class': Class,
              'Flight Distance': FlightDistance,
              'Inflight wifi service': 0,
              'Departure/Arrival time convenient': 0,
              'Ease of Online booking': 0,
              'Gate location': 0,
              'Food and drink': Food_and_drink,
              'Online boarding': Onlineboarding,
              'Seat comfort': Seat_comfort,
              'Inflight entertainment': Inflight_entertainment,
              'On-board service': On_board_service,
              'Leg room service': Leg_room_service,
              'Baggage handling': Baggage_handling,
              'Checkin service': Checkin_service,
              'Inflight service': Inflight_service,
              'Cleanliness': Cleanliness,
              'Arrival Delay in Minutes': 0,
              'Departure Delay in Minutes': 0,
              }
    features = pd.DataFrame(datacl, index=[0])
    return features
raw= pd.read_csv('train.csv')
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
# features = st.sidebar.selectbox('Features',('Raw Features',''))
features = st.sidebar.selectbox('Features',('Raw Features',))
st.sidebar.markdown("---")
HeatMatrixNaiveBayes = st.sidebar.button(label="Confusion Matrix Naive Bayes")
st.sidebar.markdown("---")
HeatMatrixDecisionTree = st.sidebar.button(label="Confusion Matrix Decision Tree")
DecisionTree = st.sidebar.button(label="Decision Tree")
st.sidebar.markdown("---")
Kmean_bieu_do = st.sidebar.button(label="K-means (Biểu đồ)");
st.sidebar.markdown("---")
Importances = st.sidebar.button(label = 'Importances Features')
st.sidebar.markdown("---")
Result = st.sidebar.button(label='Chart Results')
st.sidebar.markdown("---")

st.sidebar.subheader("Visualization Setting")
st.sidebar.write("#### Chart with Countplot")
x_label_c=st.sidebar.selectbox("X label",('Customer Type','Age','Type of Travel','Class','Flight Distance',
                                'Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service',
                                'Leg room service','Baggage handling','Checkin service','Inflight service','Cleanliness'))
hue_c=st.sidebar.selectbox("Hue",('Customer Type','Type of Travel','Class','satisfaction'))
kind_c = st.sidebar.selectbox('Kind Chart',('bar','violin','point'))
Visualize_count = st.sidebar.button(label="Visualize Countplot")
Visualize_cat_c = st.sidebar.button(label='Visualize Catplot 2 attribute')
if Visualize_count:
    st.write('''
    ## Biểu đồ biểu diễn đặc điểm các thuộc tính trong tập dữ liệu
    ''')
    st.pyplot(vs.ChartCountPlot(raw,x_label_c,hue_c))
if Visualize_cat_c:
    st.write('''
        ## Biểu đồ biểu diễn đặc điểm các thuộc tính trong tập dữ liệu
        ''')
    st.pyplot(vs.Chartcatplot2atrribute(raw, x_label_c, hue_c,kind_c))
st.sidebar.write("#### Chart with Catplot")
x_label_cat=st.sidebar.selectbox("X labels",('Customer Type','Age','Type of Travel','Class','Flight Distance',
                                'Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service',
                                'Leg room service','Baggage handling','Checkin service','Inflight service','Cleanliness'))
y_label_cat=st.sidebar.selectbox("Y labels",('Customer Type','Age','Type of Travel','Class','Flight Distance',
                                'Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service',
                                'Leg room service','Baggage handling','Checkin service','Inflight service','Cleanliness'))
hue_cat='satisfaction'
kind_cat = st.sidebar.selectbox('Kind',('bar','violin'))
col_cat=st.sidebar.selectbox("Columns",('Gender','Customer Type','Type of Travel','Class'))
Visualize_cat = st.sidebar.button(label='Visualize Catplot 3 attribute')
if Visualize_cat:
    st.write('#Biểu đồ thống kê sự hài lòng của khách hàng với 3 thuộc tính')
    st.pyplot(vs.ChartCatPlot(raw,x_label_cat,y_label_cat,hue_cat,kind_cat,col_cat))
st.sidebar.markdown("---")
mlAlgorithm = st.sidebar.selectbox('Data Mining Algorithm',('Logistic Regression','Decision Tree','Random Forest'))

#load saved classification model

data_importances = pickle.load(open('data1_importances.pkl','rb'))
logis = pickle.load(open('D:/UIT/KhaiThacDuLieu/KhaiThacDuLieu/logisraw.pkl','rb'))
logisPCA = pickle.load(open('D:/UIT/KhaiThacDuLieu/KhaiThacDuLieu/logisPCA.pkl','rb'))
tree = pickle.load(open('D:/UIT/KhaiThacDuLieu/KhaiThacDuLieu/treeraw.pkl','rb'))
treePCA = pickle.load(open('D:/UIT/KhaiThacDuLieu/KhaiThacDuLieu/treePCA.pkl','rb'))
random = pickle.load(open('D:/UIT/KhaiThacDuLieu/KhaiThacDuLieu/randomraw.pkl','rb'))
randomPCA = pickle.load(open('D:/UIT/KhaiThacDuLieu/KhaiThacDuLieu/randomPCA.pkl','rb'))

data = user_input_features()
dataraw = pd.read_csv('data_clean.csv')
dataraw = pre.ReadNDrop(dataraw)
data = pd.concat([data, dataraw], axis=0)
data = pre.TranformImport(data)
data_pca = pre.PCAImport(data)
data = data[:1]
data_pca = data_pca[:1]

Predict = st.sidebar.button(label='Predict')
st.sidebar.markdown("---")
FlightDistanceCluster = st.sidebar.slider('Flight Distance Cluster', 31, 4983)
AgeCluster = st.sidebar.slider('Age Cluster', 7, 85)

def userClusterInput():

    dataComparedCluster = {
        'age': AgeCluster,
        'fightDistance': FlightDistanceCluster,
    }
    features = pd.DataFrame(dataComparedCluster, index=[0])
    return features

Kohonen = st.sidebar.button(label="Kohonen")
Kmeans_cum = st.sidebar.button(label="K-means")




if features == 'Raw Features':
    if mlAlgorithm == 'Logistic Regression':
        prediction_proba = logis.predict(data)
    elif mlAlgorithm == 'Decision Tree':
        prediction_proba = tree.predict(data)
    elif mlAlgorithm=='Random Forest':
        prediction_proba = random.predict(data)
# elif features == 'PCA':
#     if mlAlgorithm == 'Logistic Regression':
#         prediction_proba = logisPCA.predict(data_pca)
#     elif mlAlgorithm == 'Decision Tree':
#         prediction_proba = treePCA.predict(data_pca)
#     else:
#         prediction_proba = randomPCA.predict(data_pca)
# Visualization Result
X_train,X_test,y_train,y_test = pickle.load(open('D:/UIT/KhaiThacDuLieu/KhaiThacDuLieu/data_split.pkl','rb'))
algorithm = ['LR','DT','RF']
pca = PCA(n_components=23)
X_test_pca = pca.fit_transform(X_test)

# Thac mac Thao
if Result:
    st.pyplot(vs.Chart(logis, tree, random, X_test, y_test, algorithm))


if Predict:
    st.write(prediction_proba)
if Importances:
    st.pyplot(vs.ChartImportances(tree,random,logis,data_importances))
if Kmean_bieu_do:
    dataCluster = pre.PreprocessCluster()
    age = dataCluster['Age']
    distance = dataCluster['Flight Distance']
    x = pd.concat([age, distance], axis=1).values
    st.pyplot(vs.drawCluster(x,dataCluster))
    st.pyplot(vs.drawCluster1(x,dataCluster))

if DecisionTree:
    test = pd.read_csv("test.csv")
    train = pd.read_csv("train.csv")
    data = pd.concat([train,test])
    data = data.dropna()
    data.drop(['Unnamed: 0','id','Inflight wifi service','Ease of Online booking', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'], axis=1, inplace=True)
    data.drop(['Gate location','Departure/Arrival time convenient', 'Gender'], axis=1, inplace=True)
    features = data.drop('satisfaction', axis=1)
    labels = data['satisfaction']
    features_onehot = pd.get_dummies(features, columns=features.select_dtypes(exclude=['int64']).columns)
    X_train = features_onehot[:103904]
    X_test = features_onehot[103904:]
    y_train = labels[:103904]
    y_test = labels[103904:]
    clf = tree1.DecisionTreeClassifier(criterion="entropy", random_state=0)
    clf.fit(X_train, y_train)
    tree_pred = clf.predict(X_test)
    tree_score = metrics.accuracy_score(y_test, tree_pred)
    tree_cm = metrics.confusion_matrix(y_test, tree_pred)
    # In ma tran nham lan
    plt.figure(figsize=(12,12))
    result = sns.heatmap(tree_cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    title = 'Decision Tree Accuracy Score: {0}'.format(tree_score)
    plt.title(title, size = 15)

    fig, ax = plt.subplots(figsize=(50,24))
    tree1.plot_tree(clf, filled=True, fontsize=10)
    st.pyplot(plt.show())

if HeatMatrixDecisionTree:
    test = pd.read_csv("test.csv")
    train = pd.read_csv("train.csv")
    data = pd.concat([train,test])
    data = data.dropna()
    data.drop(['Unnamed: 0','id','Inflight wifi service','Ease of Online booking', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'], axis=1, inplace=True)
    data.drop(['Gate location','Departure/Arrival time convenient', 'Gender'], axis=1, inplace=True)
    features = data.drop('satisfaction', axis=1)
    labels = data['satisfaction']
    features_onehot = pd.get_dummies(features, columns=features.select_dtypes(exclude=['int64']).columns)
    X_train = features_onehot[:103904]
    X_test = features_onehot[103904:]
    y_train = labels[:103904]
    y_test = labels[103904:]
    clf = tree1.DecisionTreeClassifier(criterion="entropy", random_state=0)
    clf.fit(X_train, y_train)
    tree_pred = clf.predict(X_test)
    tree_score = metrics.accuracy_score(y_test, tree_pred)
    tree_cm = metrics.confusion_matrix(y_test, tree_pred)

    # In ma tran nham lan
    fig = plt.figure(figsize=(12,12))
    sns.heatmap(tree_cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    title = 'Decision Tree Accuracy Score: {0}'.format(tree_score)
    plt.title(title, size = 15)
    st.pyplot(fig)

if HeatMatrixNaiveBayes:
    test = pd.read_csv("test.csv")
    train = pd.read_csv("train.csv")
    data = pd.concat([train,test])
    data = data.dropna()
    data.drop(['Unnamed: 0','id','Inflight wifi service','Ease of Online booking', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'], axis=1, inplace=True)
    data.drop(['Gate location','Departure/Arrival time convenient', 'Gender'], axis=1, inplace=True)
    features = data.drop('satisfaction', axis=1)
    labels = data['satisfaction']
    features_onehot = pd.get_dummies(features, columns=features.select_dtypes(exclude=['int64']).columns)
    X_train = features_onehot[:103904]
    X_test = features_onehot[103904:]
    y_train = labels[:103904]
    y_test = labels[103904:]

    gnb = GaussianNB()
    bayes_pred = gnb.fit(X_train, y_train).predict(X_test)
    bayes_score = metrics.accuracy_score(y_test, bayes_pred)
    bayes_cm = metrics.confusion_matrix(y_test, bayes_pred)

    figBayes = plt.figure(figsize=(12,12))
    sns.heatmap(bayes_cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Greens');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    title = 'Naive Bayes Accuracy Score: {0}'.format(bayes_score)
    plt.title(title, size = 15)
    st.pyplot(figBayes)

if Kohonen:
    dataUserClusterInput = userClusterInput()
    fightData = str(dataUserClusterInput['fightDistance'].values[0])
    ageData = str(dataUserClusterInput['age'].values[0])
    dataKohonen = pre.PreprocessCluster()
    cluster = koh.kohomen(dataKohonen, float(ageData), float(fightData))
    st.write('Thuộc cụm: '+str(cluster))


if Kmeans_cum:
    dataUserClusterInput = userClusterInput()
    fightData = float(dataUserClusterInput['fightDistance'].values[0])
    ageData = float(dataUserClusterInput['age'].values[0])
    s = [ageData, fightData]

    dataCluster = pre.PreprocessCluster()
    ageKmeans = dataCluster['Age']
    distanceKmeans = dataCluster['Flight Distance']
    x = pd.concat([ageKmeans, distanceKmeans], axis=1).values
    centroidsMain = vs.drawCluster2(x,dataCluster)
    J = koh.Kmeans(centroidsMain, s)
    st.write('Thuộc cụm: '+str(J))




