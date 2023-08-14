
from .train_model import load_data

from .rfc_all import rfc,rfc_important_feature,rfc_shap_plt
from .logistic_reg import lr,lr_if,lr_Shap
from .XGboost import XGBoost,XG_if,XG_Shap


def predictdata(file_inp):

    X1,get_trained_data = load_data(file_inp)


    clr_rf,X_train,image_data_rfc = rfc(get_trained_data)
    image_data_rfc_imp = rfc_important_feature(clr_rf=clr_rf,X_train=X_train)
    image_data_rfc_shap = rfc_shap_plt(clr_rf=clr_rf,X_train=X_train)


    clf_lr,logistic_image_data = lr(get_trained_data)

    logistic_imp_image_data = lr_if(clr_lr=clf_lr,X1=X1)

    logistic_shap_image_data = lr_Shap(clr_lr=clf_lr,data=get_trained_data)

    clr_xg,xg_image_data = XGBoost(data=get_trained_data)

    xg_imp_image_data = XG_if(clr_xg=clr_xg,data=get_trained_data)

    xg_shap_image_data = XG_Shap(clr_xg=clr_xg,data=get_trained_data)

    
    plots_data = [
        [
        {"image": image_data_rfc, "title": "Random Forest Classifier","Boxtitle":"HeatMap"},
        {"image": logistic_image_data, "title": "Logistic Regression "},
        {"image": xg_image_data, "title": "XGBOOST "},
        ],
        [
    
        {"image": image_data_rfc_imp, "title": "Random Forest Classifier","Boxtitle":"Feature Importance"},
        {"image": logistic_imp_image_data, "title": "Logistic Regression"},
        {"image": xg_imp_image_data, "title": "XGBOOST"},
        ],
        [
        
        {"image": image_data_rfc_shap, "title": "Random Forest Classifier","Boxtitle":"SHAP Summary Plot"},    
        {"image": logistic_shap_image_data, "title": "Logistic Regression "},
        {"image": xg_shap_image_data, "title": "XGBOOST"}
        ]
    ]
    return plots_data



