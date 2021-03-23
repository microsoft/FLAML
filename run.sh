

# ###run exp for 1m on all 39 datasets
# # 39*5*3 jobs / 25 cores = 24 batches 
# ## estimated duration for 1m exp: 24*1m= 24m
# python gene_script.py  -l xgb_cat -t 60.0  -m Optuna CFO BlendSearch+Optuna -filename xgb_cat_all_1m.sh -f 0 1 2 3 4 

# bash xgb_cat_all_1m.sh

# python gene_script.py  -l xgb_cat -t 60.0  -m Optuna CFO BlendSearch+Optuna -filename xgb_cat_all_1m_plot.sh -agg -plot_only

# bash xgb_cat_all_1m_plot.sh



# ###run exp for 10m on all 39 datasets
# # 39*5*3 jobs / 25 cores = 24 batches 
# ## estimated duration for 1m exp: 24*10m = 240m = 4h
python gene_script.py  -l xgb_cat -t 600.0  -m Optuna CFO BlendSearch+Optuna -filename xgb_cat_all_10m.sh -f 0 1 2 3 4 

bash xgb_cat_all_10m.sh

python gene_script.py  -l xgb_cat -t 600.0  -m Optuna CFO BlendSearch+Optuna -filename xgb_cat_all_10m_plot.sh -agg -plot_only -noredirect

xgb_cat_all_10m_plot.sh





###run exp for 10m on all 39 datasets
# 39*5*3 jobs / 25 cores = 24 batches 
## estimated duration for 1m exp: 24*10m = 240m = 4h *(2/3) = 3h
python gene_script.py  -l xgb_cat -t 600.0  -m  CFO BlendSearch+Optuna -filename xgb_cat_all_10m.sh -f 0 1 2 3 4 

bash xgb_cat_all_10m.sh

python gene_script.py  -l xgb_cat -t 600.0  -m  CFO BlendSearch+Optuna -filename xgb_cat_all_10m_plot.sh -agg -plot_only 

bash xgb_cat_all_10m_plot.sh

python gene_script.py  -l xgb_cat -t 3600.0  -m  CFO BlendSearch+Optuna -filename xgb_cat_all_1h.sh -f 0 1 2 3 4 

bash xgb_cat_all_1h.sh

# python gene_script.py  -l xgb_cat -t 3600.0  -m  CFO BlendSearch+Optuna -filename xgb_cat_all_1h_test.sh -f 0 1 2 3 4  -d KDDCup09  Airlines

# bash xgb_cat_all_1h.sh

python gene_script.py  -l xgb_cat -t 3600.0  -m  CFO BlendSearch+Optuna -filename xgb_cat_all_1h_plot.sh -agg -plot_only 

bash xgb_cat_all_1h_plot.sh

python gene_script.py  -l xgb_cat -t 60.0  -m  CFO BlendSearch+Optuna Optuna -filename xgb_cat_all_1m_plot.sh -agg -plot_only 


python gene_script.py  -l xgb_cat -t 3600.0  -m  Optuna -filename xgb_cat_all_1h_optuna.sh -f 0 1 2 3 4 

bash xgb_cat_all_1h_optuna.sh


python gene_script.py  -l xgb_cat -t 600.0  -m  Dragonfly SkOpt Nevergrad ZOOpt Ax HyperOpt -filename xgb_cat_all.sh -f 0 




python test/test_automl_exp.py  -t 600.0 -l xgb_cat -d KDDCup09  -m Dragonfly SkOpt Nevergrad ZOOpt Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 600.0 -total_pu 1 -trial_pu 1  -agg -plot_only 


python test/test_automl_exp.py  -t 600.0 -l xgb_cat -d KDDCup09  -m  SkOpt Nevergrad  Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 600.0 -total_pu 1 -trial_pu 1  -agg -plot_only 

python test/test_automl_exp.py  -t 600.0 -l lgbm -d KDDCup09  -m  SkOpt Nevergrad  Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 600.0 -total_pu 1 -trial_pu 1  -agg -plot_only 



python test/test_automl_exp.py  -t 600.0 -l xgb_blendsearch xgb_cfo xgb_hpolib  -d blood  -m  Nevergrad  Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 60.0 -total_pu 1 -trial_pu 1



python gene_script.py  -l xgb_cfo xgb_hpolib  xgb_blendsearch -t 60.0  -m  Nevergrad  Ax HyperOpt CFO BlendSearch+Optuna Optuna -filename xgb_all.sh -f 0  -d blood

python gene_script.py  -l xgb_cfo xgb_hpolib xgb_blendsearch -t 60.0  -m  Nevergrad  Ax HyperOpt CFO BlendSearch+Optuna Optuna -filename xgb_all_plot.sh -f 0  -d blood -agg -plot_only



python gene_script.py  -l xgb_cfo xgb_hpolib  xgb_blendsearch -t 3600.0  -m  Nevergrad  Ax HyperOpt CFO BlendSearch+Optuna Optuna -filename xgb_small.sh -f 0 -dlist small -max_job 48
 
python gene_script.py  -l xgb_cfo xgb_hpolib  xgb_blendsearch -t 14400.0  -m  Nevergrad  Ax HyperOpt CFO BlendSearch+Optuna Optuna -filename xgb_large1.sh -f 0 -dlist large1 -max_job 48

python gene_script.py  -l xgb_cfo xgb_hpolib  xgb_blendsearch -t 14400.0  -m  Nevergrad  Ax HyperOpt CFO BlendSearch+Optuna Optuna -filename xgb_large2.sh -f 0 -dlist large2


python gene_script.py  -l xgb_cfo xgb_hpolib  xgb_blendsearch -t 3600.0  -m  Nevergrad  Ax HyperOpt CFO BlendSearch+Optuna Optuna -filename xgb_small_plot.sh -f 0 -dlist small -max_job 48 -agg -plot_only
 
python gene_script.py  -l xgb_cfo xgb_hpolib  xgb_blendsearch -t 14400.0  -m  Nevergrad  Ax HyperOpt CFO BlendSearch+Optuna Optuna -filename xgb_large1_plot.sh -f 0 -dlist large1 -max_job 48 -agg -plot_only

python gene_script.py  -l xgb_cfo xgb_hpolib  xgb_blendsearch -t 14400.0  -m  Nevergrad  Ax HyperOpt CFO BlendSearch+Optuna Optuna -filename xgb_large2_plot.sh -f 0 -dlist large2 -agg -plot_only


python gene_script.py  -l xgb_cfo xgb_cfo_large xgb_hpolib  xgb_blendsearch  xgb_blendsearch_large lgbm_cfo lgbm_cfo_large -t 60.0  -m  Nevergrad  Ax HyperOpt CFO BlendSearch+Optuna Optuna -filename xgb_test.sh -f 1  -d blood


xgb_cfo_large xgb_blendsearch_large lgbm_cfo lgbm_cfo_large


python gene_script.py  -l lgbm_cfo lgbm_cfo_large -t 3600.0  -m  Nevergrad  Ax HyperOpt CFO BlendSearch+Optuna Optuna -filename lgbm_small.sh -f 0 -dlist small -max_job 48
 
python gene_script.py  -l lgbm_cfo lgbm_cfo_large -t 14400.0  -m  Nevergrad  Ax HyperOpt CFO BlendSearch+Optuna Optuna -filename lgbm_large1.sh -f 0 -dlist large1 -max_job 48
