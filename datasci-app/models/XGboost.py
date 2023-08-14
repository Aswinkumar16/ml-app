from .packages import *


def XGBoost(data):
    plt.figure(figsize=(8, 6))  # Set the size of the figure

    X_train = data[0]
    X_test = data[1]
    Y_train = data[2]
    Y_test = data[3]

    

    #Classification uisng XGBoost
    clf_xg = XGBClassifier()
    clr_xg = clf_xg.fit(X_train, Y_train)
    predictions_xg = clr_xg.predict(X_test)
    ac_xg = accuracy_score(Y_test, predictions_xg)
    print("Overall accuracy of Random Forest model using test-set is : %f" %(ac_xg*100))
    print(classification_report(Y_test, predictions_xg))
    cm = confusion_matrix(Y_test,clf_xg.predict(X_test))
    ax = sns.heatmap(cm,annot=True,fmt="d")
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300,bbox_inches='tight')
    buffer.seek(0)
    
    # Convert the image data to base64
    image_data = base64.b64encode(buffer.read()).decode()
    plt.clf()
    return clr_xg,image_data

def XG_if(clr_xg, data):
    X_train = data[0]
    features = X_train.columns
    
    plt.figure(figsize=(8, 6))  # Set the size of the figure
    
    # Feature Importance
    importances = clr_xg.feature_importances_
    indices = np.argsort(importances)
    
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    
    # Convert the image data to base64
    image_data = base64.b64encode(buffer.read()).decode()
    plt.clf()
    return image_data

def XG_Shap(clr_xg,data):
    plt.figure(figsize=(8, 6))  # Set the size of the figure

    X_train = data[0]
    # Shap
    shap_values = shap.TreeExplainer(clr_xg).shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar")
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300,bbox_inches='tight')
    buffer.seek(0)
    
    # Convert the image data to base64
    image_data = base64.b64encode(buffer.read()).decode()
    plt.clf()
    return image_data

