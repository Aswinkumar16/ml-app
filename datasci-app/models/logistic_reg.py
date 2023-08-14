
from .packages import *


def lr(data):

    X_train = data[0]
    X_test = data[1]
    Y_train = data[2]
    Y_test = data[3]

    plt.figure(figsize=(8, 6))  # Set the size of the figure

    #Classification using Logistic Regression
    clf_lr = LogisticRegression(random_state=0)
    clr_lr = clf_lr.fit(X_train, Y_train)
    predictions_lr = clr_lr.predict(X_test)
    ac_lr = accuracy_score(Y_test, predictions_lr)
    print("Overall accuracy of Random Forest model using test-set is : %f" %(ac_lr*100))
    print(classification_report(Y_test, predictions_lr))
    cm = confusion_matrix(Y_test,clf_lr.predict(X_test))
    
    ax = sns.heatmap(cm,annot=True,fmt="d")
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300,bbox_inches='tight')
    buffer.seek(0)
    
    # Convert the image data to base64
    image_data = base64.b64encode(buffer.read()).decode()
    plt.clf()
    return clf_lr,image_data



def lr_if(clr_lr,X1):
    #Feature Importance
    coefficients = clr_lr.coef_
    avg_importance = np.mean(np.abs(coefficients), axis=0)
    feature_importance = pd.DataFrame({'Feature': X1.columns, 'Importance': avg_importance})
    feature_importance = feature_importance.sort_values('Importance', ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6.70))  # Create a figure and axis

    feature_importance.plot(x='Feature', y='Importance', kind='barh',ax=ax)
    # Set the size of the figure

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    
    # Convert the image data to base64
    image_data = base64.b64encode(buffer.read()).decode()
    
    plt.clf()
    return image_data



def lr_Shap(clr_lr,data):
    plt.figure(figsize=(8, 6))  # Set the size of the figure

    X_train = data[0]
    X_test = data[1]
    features = X_train.columns

    #Shap
    explainer = shap.Explainer(clr_lr, X_train, feature_names=features)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_train, plot_type="bar")
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300,bbox_inches='tight')
    buffer.seek(0)
    
    # Convert the image data to base64
    image_data = base64.b64encode(buffer.read()).decode()
    plt.clf()
    return image_data
