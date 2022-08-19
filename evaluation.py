import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,classification_report,confusion_matrix

def get_performance(predictions, y_test, labels=[1, 0]):

    accuracy = accuracy_score(predictions,y_test)   
    precision = precision_score(predictions,y_test)   
    recall = recall_score(predictions,y_test)   
    f1_score1 = f1_score(predictions,y_test) 
    
    report = classification_report(predictions,y_test)
    
    cm = confusion_matrix(predictions,y_test)  
    cm_as_dataframe = pd.DataFrame(data=cm)
    
    print('Model Performance metrics:')
    print('-'*30)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1_score1)
    print('\nModel Classification report:')
    print('-'*30)
    print(report)
    print('\nPrediction Confusion Matrix:')
    print('-'*30)
    print(cm_as_dataframe)
    
    return accuracy, precision, recall, f1_score1


def plot_roc(model, y_test, features):
   
    fpr, tpr, _= metrics.roc_curve(y_test,model.predict_proba(features)[:,1]) 
    roc_auc = metrics.auc(fpr,tpr) 

    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc})', linewidth=2.5)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc