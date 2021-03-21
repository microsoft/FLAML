
screen -Sdm agg-bloo-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d blood  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-bloo-xgb_cfo-Neve-0 2>./stdout/err_agg-bloo-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm agg-bloo-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d blood  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-bloo-xgb_hpolib-Neve-0 2>./stdout/err_agg-bloo-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm agg-bloo-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d blood  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-bloo-xgb_blendsearch-Neve-0 2>./stdout/err_agg-bloo-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm agg-Aust-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d Australian  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-Aust-xgb_cfo-Neve-0 2>./stdout/err_agg-Aust-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm agg-Aust-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d Australian  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-Aust-xgb_hpolib-Neve-0 2>./stdout/err_agg-Aust-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm agg-Aust-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d Australian  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-Aust-xgb_blendsearch-Neve-0 2>./stdout/err_agg-Aust-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm agg-cred-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d credit  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-cred-xgb_cfo-Neve-0 2>./stdout/err_agg-cred-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm agg-cred-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d credit  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-cred-xgb_hpolib-Neve-0 2>./stdout/err_agg-cred-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm agg-cred-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d credit  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-cred-xgb_blendsearch-Neve-0 2>./stdout/err_agg-cred-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm agg-car-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d car  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-car-xgb_cfo-Neve-0 2>./stdout/err_agg-car-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm agg-car-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d car  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-car-xgb_hpolib-Neve-0 2>./stdout/err_agg-car-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm agg-car-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d car  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-car-xgb_blendsearch-Neve-0 2>./stdout/err_agg-car-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm agg-kc1-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d kc1  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-kc1-xgb_cfo-Neve-0 2>./stdout/err_agg-kc1-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm agg-kc1-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d kc1  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-kc1-xgb_hpolib-Neve-0 2>./stdout/err_agg-kc1-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm agg-kc1-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d kc1  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-kc1-xgb_blendsearch-Neve-0 2>./stdout/err_agg-kc1-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm agg-kr-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d kr  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-kr-xgb_cfo-Neve-0 2>./stdout/err_agg-kr-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm agg-kr-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d kr  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-kr-xgb_hpolib-Neve-0 2>./stdout/err_agg-kr-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm agg-kr-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d kr  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-kr-xgb_blendsearch-Neve-0 2>./stdout/err_agg-kr-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm agg-phon-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d phoneme  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-phon-xgb_cfo-Neve-0 2>./stdout/err_agg-phon-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm agg-phon-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d phoneme  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-phon-xgb_hpolib-Neve-0 2>./stdout/err_agg-phon-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm agg-phon-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d phoneme  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-phon-xgb_blendsearch-Neve-0 2>./stdout/err_agg-phon-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm agg-segm-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d segment  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-segm-xgb_cfo-Neve-0 2>./stdout/err_agg-segm-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm agg-segm-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d segment  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-segm-xgb_hpolib-Neve-0 2>./stdout/err_agg-segm-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm agg-segm-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d segment  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-segm-xgb_blendsearch-Neve-0 2>./stdout/err_agg-segm-xgb_blendsearch-Neve-0"
sleep 10s

screen -Sdm agg-bank-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d bank  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-bank-xgb_cfo-Neve-0 2>./stdout/err_agg-bank-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm agg-bank-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d bank  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-bank-xgb_hpolib-Neve-0 2>./stdout/err_agg-bank-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm agg-bank-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d bank  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-bank-xgb_blendsearch-Neve-0 2>./stdout/err_agg-bank-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm agg-Albe-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d Albert  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-Albe-xgb_cfo-Neve-0 2>./stdout/err_agg-Albe-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm agg-Albe-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d Albert  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-Albe-xgb_hpolib-Neve-0 2>./stdout/err_agg-Albe-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm agg-Albe-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d Albert  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-Albe-xgb_blendsearch-Neve-0 2>./stdout/err_agg-Albe-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm agg-APSF-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d APSFailure  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-APSF-xgb_cfo-Neve-0 2>./stdout/err_agg-APSF-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm agg-APSF-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d APSFailure  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-APSF-xgb_hpolib-Neve-0 2>./stdout/err_agg-APSF-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm agg-APSF-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d APSFailure  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-APSF-xgb_blendsearch-Neve-0 2>./stdout/err_agg-APSF-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm agg-noma-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d nomao  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-noma-xgb_cfo-Neve-0 2>./stdout/err_agg-noma-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm agg-noma-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d nomao  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-noma-xgb_hpolib-Neve-0 2>./stdout/err_agg-noma-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm agg-noma-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d nomao  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-noma-xgb_blendsearch-Neve-0 2>./stdout/err_agg-noma-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm agg-nume-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d numerai28  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-nume-xgb_cfo-Neve-0 2>./stdout/err_agg-nume-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm agg-nume-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d numerai28  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-nume-xgb_hpolib-Neve-0 2>./stdout/err_agg-nume-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm agg-nume-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d numerai28  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-nume-xgb_blendsearch-Neve-0 2>./stdout/err_agg-nume-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm agg-Hele-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d Helena  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-Hele-xgb_cfo-Neve-0 2>./stdout/err_agg-Hele-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm agg-Hele-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d Helena  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-Hele-xgb_hpolib-Neve-0 2>./stdout/err_agg-Hele-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm agg-Hele-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d Helena  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-Hele-xgb_blendsearch-Neve-0 2>./stdout/err_agg-Hele-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm agg-KDDC-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d KDDCup09  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-KDDC-xgb_cfo-Neve-0 2>./stdout/err_agg-KDDC-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm agg-KDDC-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d KDDCup09  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-KDDC-xgb_hpolib-Neve-0 2>./stdout/err_agg-KDDC-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm agg-KDDC-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d KDDCup09  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-KDDC-xgb_blendsearch-Neve-0 2>./stdout/err_agg-KDDC-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm agg-adul-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d adult  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-adul-xgb_cfo-Neve-0 2>./stdout/err_agg-adul-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm agg-adul-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d adult  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-adul-xgb_hpolib-Neve-0 2>./stdout/err_agg-adul-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm agg-adul-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d adult  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-adul-xgb_blendsearch-Neve-0 2>./stdout/err_agg-adul-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm agg-Amaz-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d Amazon  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-Amaz-xgb_cfo-Neve-0 2>./stdout/err_agg-Amaz-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm agg-Amaz-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d Amazon  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-Amaz-xgb_hpolib-Neve-0 2>./stdout/err_agg-Amaz-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm agg-Amaz-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d Amazon  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-Amaz-xgb_blendsearch-Neve-0 2>./stdout/err_agg-Amaz-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm agg-vehi-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d vehicle  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-vehi-xgb_cfo-Neve-0 2>./stdout/err_agg-vehi-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm agg-vehi-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d vehicle  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-vehi-xgb_hpolib-Neve-0 2>./stdout/err_agg-vehi-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm agg-vehi-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d vehicle  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-vehi-xgb_blendsearch-Neve-0 2>./stdout/err_agg-vehi-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm agg-Dion-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d Dionis  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-Dion-xgb_cfo-Neve-0 2>./stdout/err_agg-Dion-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm agg-Dion-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d Dionis  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-Dion-xgb_hpolib-Neve-0 2>./stdout/err_agg-Dion-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm agg-Dion-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d Dionis  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-Dion-xgb_blendsearch-Neve-0 2>./stdout/err_agg-Dion-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm agg-dilb-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d dilbert  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-dilb-xgb_cfo-Neve-0 2>./stdout/err_agg-dilb-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm agg-dilb-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d dilbert  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-dilb-xgb_hpolib-Neve-0 2>./stdout/err_agg-dilb-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm agg-dilb-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d dilbert  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-dilb-xgb_blendsearch-Neve-0 2>./stdout/err_agg-dilb-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm agg-Cove-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d Covertype  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-Cove-xgb_cfo-Neve-0 2>./stdout/err_agg-Cove-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm agg-Cove-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d Covertype  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-Cove-xgb_hpolib-Neve-0 2>./stdout/err_agg-Cove-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm agg-Cove-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d Covertype  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-Cove-xgb_blendsearch-Neve-0 2>./stdout/err_agg-Cove-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm agg-Robe-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d Robert  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-Robe-xgb_cfo-Neve-0 2>./stdout/err_agg-Robe-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm agg-Robe-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d Robert  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-Robe-xgb_hpolib-Neve-0 2>./stdout/err_agg-Robe-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm agg-Robe-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d Robert  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-Robe-xgb_blendsearch-Neve-0 2>./stdout/err_agg-Robe-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm agg-Fash-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d Fashion  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-Fash-xgb_cfo-Neve-0 2>./stdout/err_agg-Fash-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm agg-Fash-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d Fashion  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-Fash-xgb_hpolib-Neve-0 2>./stdout/err_agg-Fash-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm agg-Fash-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d Fashion  -m Nevergrad Ax HyperOpt CFO BlendSearch+Optuna Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0  -plot_only  -agg >./stdout/out_agg-Fash-xgb_blendsearch-Neve-0 2>./stdout/err_agg-Fash-xgb_blendsearch-Neve-0"
sleep 10s