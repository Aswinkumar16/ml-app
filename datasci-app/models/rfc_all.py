


from .packages import *

def rfc(data):

    X_train = data[0]
    X_test = data[1]
    Y_train = data[2]
    Y_test = data[3]

    #random forest classifier with n_estimators=10 (default)
    clf_rf = RandomForestClassifier(random_state=43)
    clr_rf = clf_rf.fit(X_train,Y_train)
    predictions_rf = clr_rf.predict(X_test)
    ac_rf = accuracy_score(Y_test, predictions_rf)
    print("Overall accuracy of Random Forest model using test-set is : %f" %(ac_rf*100))
    print(classification_report(Y_test, predictions_rf))
    cm = confusion_matrix(Y_test,clf_rf.predict(X_test))
    plt.figure(figsize=(8, 6))  # Set the size of the figure

    ax = sns.heatmap(cm,annot=True,fmt="d")
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300,bbox_inches='tight')
    buffer.seek(0)
    
    # Convert the image data to base64
    image_data = base64.b64encode(buffer.read()).decode()
    #plt.savefig("./plotimg/rfc-s.png")
    plt.clf()
    return clr_rf,X_train,image_data

def rfc_important_feature(clr_rf, X_train):
    # Feature Importance
    importances = clr_rf.feature_importances_
    indices = np.argsort(importances)
    features = X_train.columns
    
    plt.figure(figsize=(8, 6))  # Set the size of the figure
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



def rfc_shap_plt(clr_rf,X_train):
    # Shap 
    plt.figure(figsize=(8, 6))  # Set the size of the figure

    shap_values = shap.TreeExplainer(clr_rf).shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar")
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300,bbox_inches='tight')
    buffer.seek(0)
    
    # Convert the image data to base64
    image_data = base64.b64encode(buffer.read()).decode()
    plt.clf()
    return image_data


