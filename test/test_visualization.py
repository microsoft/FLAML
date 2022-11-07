from flaml import AutoML
from flaml.data import load_openml_dataset
from flaml.data import get_output_from_log
import matplotlib.pyplot as plt
import statsmodels.api as sm

X_train, X_test, y_train, y_test = load_openml_dataset(dataset_id=1169, data_dir='./')
print("Data type:", type(X_train), type(y_train))

automl = AutoML()

settings = {
    "time_budget": 60,  # total running time in seconds
    "metric": 'accuracy',  # can be: 'r2', 'rmse', 'mae', 'mse', 'accuracy', 'roc_auc', 'roc_auc_ovr',
                           # 'roc_auc_ovo', 'log_loss', 'mape', 'f1', 'ap', 'ndcg', 'micro_f1', 'macro_f1'
    "task": 'classification',  # task type
    "log_file_name": 'airlines_experiment.log',  # flaml log file
    "seed": 7654321,    # random seed
}

automl.fit(X_train=X_train, y_train=y_train, **settings)

 
time_history, best_valid_loss_history, valid_loss_history, config_history, metric_history = \
    get_output_from_log(filename=settings['log_file_name'], time_budget=240)
for config in config_history:
    print(config)
print(metric_history)

# ATMSeer: increasing transparency and controllability in automated ml page 4
# Whither Automl? Understanding the role of automation in machine learning
# Which parameter is more important, which models are working the best dataset  
# XAutoML A visual analytics tools for establishing trust


t = "Learning Curve"
xl = "Wall Clock Time (s)"
yl = "Validation Accuracy"
pt = "feature"
bvlh = best_valid_loss_history
vlh = valid_loss_history
th = time_history
automl.visualization(type = "feature_importance")
automl.visualization(type = "validation_accuracy", xlab = xl, ylab = yl, settings = settings)



'''
if plottype == "scatter":
            plt.title(title)
            plt.xlabel(xlab)
            plt.ylabel(ylab)
            plt.scatter(time_history, 1 - np.array(valid_loss_history))
            plt.step(time_history, 1 - np.array(best_valid_loss_history), where='post')
            plt.show()
elif plottype == "feature":
            plt.title(title)
            plt.barh(self.feature_names_in_, self.feature_importances_)
            plt.show()
        elif plottype == "Model":
            # pie graph that shows percentage of each model with the two best
            print("best model b")
        elif plottype == "parameters":
            # ANOVA 
            print("dees the best")
'''