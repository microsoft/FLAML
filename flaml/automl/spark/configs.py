ParamList_LightGBM_Base = ["featuresCol", "labelCol", "objective"]
ParamList_LightGBM_Classifier = ParamList_LightGBM_Base + ["isUnbalance"]
ParamList_LightGBM_Regressor = ParamList_LightGBM_Base
ParamList_LightGBM_Ranker = ParamList_LightGBM_Base + ["groupCol"]
Default_groupCol = "query"
