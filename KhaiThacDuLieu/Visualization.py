import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pylab as plb
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics



raw = pd.read_csv('train.csv')


pca = PCA(n_components=23)

X_train,X_test,y_train,y_test = pickle.load(open('D:/UIT/KhaiThacDuLieu/KhaiThacDuLieu/data_split.pkl','rb'))
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
logis = pickle.load(open('D:/UIT/KhaiThacDuLieu/KhaiThacDuLieu/logisraw.pkl','rb'))
logisPCA = pickle.load(open('D:/UIT/KhaiThacDuLieu/KhaiThacDuLieu/logisPCA.pkl','rb'))
tree = pickle.load(open('D:/UIT/KhaiThacDuLieu/KhaiThacDuLieu/treeraw.pkl','rb'))
treePCA = pickle.load(open('D:/UIT/KhaiThacDuLieu/KhaiThacDuLieu/treePCA.pkl','rb'))
random = pickle.load(open('D:/UIT/KhaiThacDuLieu/KhaiThacDuLieu/randomraw.pkl','rb'))
randomPCA = pickle.load(open('D:/UIT/KhaiThacDuLieu/KhaiThacDuLieu/randomPCA.pkl','rb'))
data_importances = pickle.load(open('data1_importances.pkl','rb'))
list = [logis,tree,random]
list_pca = [logisPCA,treePCA,randomPCA]
list_kq_pca =[]
list_kq_raw = []
for i,j in zip(list,list_pca):
    list_kq_raw.append(i.score(X_test,y_test))
    list_kq_pca.append(j.score(X_test_pca,y_test))
algorithm = ['LR','DT','RF']
# Chart bieu dien so sanh cac do do cua mo hinh phan lop:
def Chart(model,model1,model2,X_test,y_test,algorithm):
    precision_logis,recall_logis,f1_logis,support_logis = precision_recall_fscore_support(y_test,model.predict(X_test))
    predict_tree,recall_tree,f1_tree,support_tree = precision_recall_fscore_support(y_test,model1.predict(X_test))
    predict_random, recall_random, f1_random, support_random = precision_recall_fscore_support(y_test,model2.predict(X_test))
    def Dodo(model1,model2,model3,algorithm):
        list0 = [model1[0],model2[0],model3[0]]
        list1 = [model1[1],model2[1],model3[1]]
        index = np.arange(3)
        width = 0.2
        plt.bar(index,list0,width=width,color='blue',label='Neutral')
        plt.bar(index+width, list1, width=width, color='green', label='Satisfied')
        plt.ylabel("Accuracy")
        plt.ylim(0,1.1)
        plt.xticks(index+width/2,algorithm)
        plt.legend(loc=2)
    plt.figure(figsize=(12,6))
    plt.subplot(1,3,1)
    Dodo(precision_logis,predict_tree,predict_random,algorithm)
    plt.xlabel("Precision")
    plt.subplot(1,3,2)
    Dodo(recall_logis,recall_tree,recall_random,algorithm)
    plt.xlabel("Recall")
    plt.title("Metrics Chart Comparation(Precision - Recall - F1-Score)")
    plt.subplot(1,3,3)
    Dodo(f1_logis,f1_tree,f1_random,algorithm)
    plt.xlabel("F1-Score")
    plt.show()
# Chart bieu dien Importance Features
def ChartImportances(model1,model2,model3,data):
    def ImportanceFeatures(model, data):
        importances = model.feature_importances_
        feature = np.array(data.columns)
        factor = np.argsort(importances)
        plt.barh(range(len(factor)), importances[factor], color='b', align='center')
        plt.yticks(range(len(factor)), feature[factor])
    def ImportanceFeatures_logis(model,data):
        importances = model.coef_.reshape(-1)
        feature = np.array(data.columns)
        factor = np.argsort(importances)
        plb.barh(range(len(factor)),importances[factor],color='blue',align='center')
        plt.yticks(range(len(factor)),feature[factor])
    plt.figure(figsize=(32,12))
    plt.subplot(2,3,1)
    plt.subplots_adjust(hspace=0.5, wspace = 0.5) #khoang cach
    ImportanceFeatures(model1,data)
    plt.xlabel("Decision Tree")
    plt.subplot(2,3,2)
    ImportanceFeatures(model2,data)
    plt.title("Importance Features of each Algorithm")
    plt.xlabel("Random Forest")
    plt.subplot(2,3,4)
    ImportanceFeatures_logis(model3,data)
    plt.xlabel("Logistic Regression")
    plt.show()
#ChartImportances(tree,random,logis,data_importances)
def ChartCountPlot(data,x_axis,hue):
    plt.figure(figsize=(30,20))
    sns.countplot(x_axis,data=data,hue=hue)
    plt.title("Chart of "+x_axis +" by "+hue)
    plt.show()
def Chartcatplot2atrribute(data,x_axis,y_axis,kind_c):
    data['satisfaction'] = [1 if each == 'satisfied' else 0 for each in data['satisfaction']]
    sns.catplot(x_axis,y_axis, data=data,kind=kind_c,aspect=5,height=5)
    plt.title("Chart of " + x_axis + " and " +y_axis )
    plt.show()
Chartcatplot2atrribute(raw,'Age','satisfaction','point')

def ChartCatPlot(data,x_axis,y_axis,hue,kind,col):
    plt.figure(figsize=(20, 10))
    sns.catplot(x=x_axis,y=y_axis,data=data,kind=kind,hue=hue,col=col,legend_out=False)
    plt.show()


def drawCluster(x, data):
    Distortion = []
    K = range(1,11)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(x)
        Distortion.append(kmeanModel.inertia_)

    plt.figure(1, figsize=(15,6))
    plt.plot(K, Distortion, 'o')
    plt.plot(K, Distortion, '-', alpha= 0.5)
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('Distortion of age and score')
    plt.show()



def drawCluster1(x,data):
    kmeanModel = KMeans(n_clusters = 3)#, init = 'random')
    kmeanModel.fit(x)
    labels = kmeanModel.labels_ #danh sach các điểm thuộc cụm nào
    centroids = kmeanModel.cluster_centers_ #tâm cụm
    centroidsGlo = centroids
    age = data['Age']
    distance = data['Flight Distance']
    #kết quả gom cụm băng biểu đồ 
    plt.figure(1, figsize=(15,7))
    plt.clf()
    plt.scatter(x=age, y = distance, c=labels, s= 100)
    plt.scatter(x=centroids[:, 0], y = centroids[:, 1], s= 100, c='red', alpha=0.5)
    plt.ylabel('Spending Score (1-100)')
    plt.xlabel('Age')
    plt.show()

def drawCluster2(x,data):
    kmeanModel = KMeans(n_clusters = 3)#, init = 'random')
    kmeanModel.fit(x)
    labels = kmeanModel.labels_ #danh sach các điểm thuộc cụm nào
    centroids = kmeanModel.cluster_centers_ #tâm cụm
    age = data['Age']
    distance = data['Flight Distance']
    #kết quả gom cụm băng biểu đồ 
    plt.figure(1, figsize=(15,7))
    plt.clf()
    plt.scatter(x=age, y = distance, c=labels, s= 100)
    plt.scatter(x=centroids[:, 0], y = centroids[:, 1], s= 100, c='red', alpha=0.5)
    plt.ylabel('Spending Score (1-100)')
    plt.xlabel('Age')
    return centroids



