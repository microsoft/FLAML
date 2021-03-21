
screen -Sdm bloo-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d blood  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-xgb_cfo-Neve-0 2>./stdout/err_bloo-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm bloo-xgb_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d blood  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-xgb_cfo-Ax-0 2>./stdout/err_bloo-xgb_cfo-Ax-0"
sleep 10s
screen -Sdm bloo-xgb_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d blood  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-xgb_cfo-Hype-0 2>./stdout/err_bloo-xgb_cfo-Hype-0"
sleep 10s
screen -Sdm bloo-xgb_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d blood  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-xgb_cfo-CFO-0 2>./stdout/err_bloo-xgb_cfo-CFO-0"
sleep 10s
screen -Sdm bloo-xgb_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d blood  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-xgb_cfo-BS+O-0 2>./stdout/err_bloo-xgb_cfo-BS+O-0"
sleep 10s
screen -Sdm bloo-xgb_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d blood  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-xgb_cfo-Optu-0 2>./stdout/err_bloo-xgb_cfo-Optu-0"
sleep 10s
screen -Sdm bloo-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d blood  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-xgb_hpolib-Neve-0 2>./stdout/err_bloo-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm bloo-xgb_hpolib-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d blood  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-xgb_hpolib-Ax-0 2>./stdout/err_bloo-xgb_hpolib-Ax-0"
sleep 10s
screen -Sdm bloo-xgb_hpolib-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d blood  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-xgb_hpolib-Hype-0 2>./stdout/err_bloo-xgb_hpolib-Hype-0"
sleep 10s
screen -Sdm bloo-xgb_hpolib-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d blood  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-xgb_hpolib-CFO-0 2>./stdout/err_bloo-xgb_hpolib-CFO-0"
sleep 10s
screen -Sdm bloo-xgb_hpolib-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d blood  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-xgb_hpolib-BS+O-0 2>./stdout/err_bloo-xgb_hpolib-BS+O-0"
sleep 10s
screen -Sdm bloo-xgb_hpolib-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d blood  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-xgb_hpolib-Optu-0 2>./stdout/err_bloo-xgb_hpolib-Optu-0"
sleep 10s
screen -Sdm bloo-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d blood  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-xgb_blendsearch-Neve-0 2>./stdout/err_bloo-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm bloo-xgb_blendsearch-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d blood  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-xgb_blendsearch-Ax-0 2>./stdout/err_bloo-xgb_blendsearch-Ax-0"
sleep 10s
screen -Sdm bloo-xgb_blendsearch-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d blood  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-xgb_blendsearch-Hype-0 2>./stdout/err_bloo-xgb_blendsearch-Hype-0"
sleep 10s
screen -Sdm bloo-xgb_blendsearch-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d blood  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-xgb_blendsearch-CFO-0 2>./stdout/err_bloo-xgb_blendsearch-CFO-0"
sleep 10s
screen -Sdm bloo-xgb_blendsearch-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d blood  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-xgb_blendsearch-BS+O-0 2>./stdout/err_bloo-xgb_blendsearch-BS+O-0"
sleep 10s
screen -Sdm bloo-xgb_blendsearch-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d blood  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-xgb_blendsearch-Optu-0 2>./stdout/err_bloo-xgb_blendsearch-Optu-0"
sleep 10s
screen -Sdm Aust-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d Australian  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-xgb_cfo-Neve-0 2>./stdout/err_Aust-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm Aust-xgb_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d Australian  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-xgb_cfo-Ax-0 2>./stdout/err_Aust-xgb_cfo-Ax-0"
sleep 10s
screen -Sdm Aust-xgb_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d Australian  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-xgb_cfo-Hype-0 2>./stdout/err_Aust-xgb_cfo-Hype-0"
sleep 10s
screen -Sdm Aust-xgb_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d Australian  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-xgb_cfo-CFO-0 2>./stdout/err_Aust-xgb_cfo-CFO-0"
sleep 10s
screen -Sdm Aust-xgb_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d Australian  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-xgb_cfo-BS+O-0 2>./stdout/err_Aust-xgb_cfo-BS+O-0"
sleep 10s
screen -Sdm Aust-xgb_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d Australian  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-xgb_cfo-Optu-0 2>./stdout/err_Aust-xgb_cfo-Optu-0"
sleep 10s
screen -Sdm Aust-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d Australian  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-xgb_hpolib-Neve-0 2>./stdout/err_Aust-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm Aust-xgb_hpolib-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d Australian  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-xgb_hpolib-Ax-0 2>./stdout/err_Aust-xgb_hpolib-Ax-0"
sleep 10s
screen -Sdm Aust-xgb_hpolib-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d Australian  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-xgb_hpolib-Hype-0 2>./stdout/err_Aust-xgb_hpolib-Hype-0"
sleep 10s
screen -Sdm Aust-xgb_hpolib-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d Australian  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-xgb_hpolib-CFO-0 2>./stdout/err_Aust-xgb_hpolib-CFO-0"
sleep 10s
screen -Sdm Aust-xgb_hpolib-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d Australian  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-xgb_hpolib-BS+O-0 2>./stdout/err_Aust-xgb_hpolib-BS+O-0"
sleep 10s
screen -Sdm Aust-xgb_hpolib-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d Australian  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-xgb_hpolib-Optu-0 2>./stdout/err_Aust-xgb_hpolib-Optu-0"
sleep 10s
screen -Sdm Aust-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d Australian  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-xgb_blendsearch-Neve-0 2>./stdout/err_Aust-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm Aust-xgb_blendsearch-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d Australian  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-xgb_blendsearch-Ax-0 2>./stdout/err_Aust-xgb_blendsearch-Ax-0"
sleep 10s
screen -Sdm Aust-xgb_blendsearch-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d Australian  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-xgb_blendsearch-Hype-0 2>./stdout/err_Aust-xgb_blendsearch-Hype-0"
sleep 10s
screen -Sdm Aust-xgb_blendsearch-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d Australian  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-xgb_blendsearch-CFO-0 2>./stdout/err_Aust-xgb_blendsearch-CFO-0"
sleep 10s
screen -Sdm Aust-xgb_blendsearch-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d Australian  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-xgb_blendsearch-BS+O-0 2>./stdout/err_Aust-xgb_blendsearch-BS+O-0"
sleep 10s
screen -Sdm Aust-xgb_blendsearch-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d Australian  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-xgb_blendsearch-Optu-0 2>./stdout/err_Aust-xgb_blendsearch-Optu-0"
sleep 10s
screen -Sdm cred-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d credit  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-xgb_cfo-Neve-0 2>./stdout/err_cred-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm cred-xgb_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d credit  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-xgb_cfo-Ax-0 2>./stdout/err_cred-xgb_cfo-Ax-0"
sleep 10s
screen -Sdm cred-xgb_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d credit  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-xgb_cfo-Hype-0 2>./stdout/err_cred-xgb_cfo-Hype-0"
sleep 10s
screen -Sdm cred-xgb_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d credit  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-xgb_cfo-CFO-0 2>./stdout/err_cred-xgb_cfo-CFO-0"
sleep 10s
screen -Sdm cred-xgb_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d credit  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-xgb_cfo-BS+O-0 2>./stdout/err_cred-xgb_cfo-BS+O-0"
sleep 10s
screen -Sdm cred-xgb_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d credit  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-xgb_cfo-Optu-0 2>./stdout/err_cred-xgb_cfo-Optu-0"
sleep 10s
screen -Sdm cred-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d credit  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-xgb_hpolib-Neve-0 2>./stdout/err_cred-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm cred-xgb_hpolib-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d credit  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-xgb_hpolib-Ax-0 2>./stdout/err_cred-xgb_hpolib-Ax-0"
sleep 10s
screen -Sdm cred-xgb_hpolib-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d credit  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-xgb_hpolib-Hype-0 2>./stdout/err_cred-xgb_hpolib-Hype-0"
sleep 10s
screen -Sdm cred-xgb_hpolib-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d credit  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-xgb_hpolib-CFO-0 2>./stdout/err_cred-xgb_hpolib-CFO-0"
sleep 10s
screen -Sdm cred-xgb_hpolib-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d credit  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-xgb_hpolib-BS+O-0 2>./stdout/err_cred-xgb_hpolib-BS+O-0"
sleep 10s
screen -Sdm cred-xgb_hpolib-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d credit  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-xgb_hpolib-Optu-0 2>./stdout/err_cred-xgb_hpolib-Optu-0"
sleep 10s
sleep 3960s
screen -Sdm cred-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d credit  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-xgb_blendsearch-Neve-0 2>./stdout/err_cred-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm cred-xgb_blendsearch-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d credit  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-xgb_blendsearch-Ax-0 2>./stdout/err_cred-xgb_blendsearch-Ax-0"
sleep 10s
screen -Sdm cred-xgb_blendsearch-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d credit  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-xgb_blendsearch-Hype-0 2>./stdout/err_cred-xgb_blendsearch-Hype-0"
sleep 10s
screen -Sdm cred-xgb_blendsearch-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d credit  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-xgb_blendsearch-CFO-0 2>./stdout/err_cred-xgb_blendsearch-CFO-0"
sleep 10s
screen -Sdm cred-xgb_blendsearch-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d credit  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-xgb_blendsearch-BS+O-0 2>./stdout/err_cred-xgb_blendsearch-BS+O-0"
sleep 10s
screen -Sdm cred-xgb_blendsearch-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d credit  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-xgb_blendsearch-Optu-0 2>./stdout/err_cred-xgb_blendsearch-Optu-0"
sleep 10s
screen -Sdm car-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d car  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-xgb_cfo-Neve-0 2>./stdout/err_car-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm car-xgb_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d car  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-xgb_cfo-Ax-0 2>./stdout/err_car-xgb_cfo-Ax-0"
sleep 10s
screen -Sdm car-xgb_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d car  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-xgb_cfo-Hype-0 2>./stdout/err_car-xgb_cfo-Hype-0"
sleep 10s
screen -Sdm car-xgb_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d car  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-xgb_cfo-CFO-0 2>./stdout/err_car-xgb_cfo-CFO-0"
sleep 10s
screen -Sdm car-xgb_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d car  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-xgb_cfo-BS+O-0 2>./stdout/err_car-xgb_cfo-BS+O-0"
sleep 10s
screen -Sdm car-xgb_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d car  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-xgb_cfo-Optu-0 2>./stdout/err_car-xgb_cfo-Optu-0"
sleep 10s
screen -Sdm car-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d car  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-xgb_hpolib-Neve-0 2>./stdout/err_car-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm car-xgb_hpolib-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d car  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-xgb_hpolib-Ax-0 2>./stdout/err_car-xgb_hpolib-Ax-0"
sleep 10s
screen -Sdm car-xgb_hpolib-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d car  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-xgb_hpolib-Hype-0 2>./stdout/err_car-xgb_hpolib-Hype-0"
sleep 10s
screen -Sdm car-xgb_hpolib-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d car  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-xgb_hpolib-CFO-0 2>./stdout/err_car-xgb_hpolib-CFO-0"
sleep 10s
screen -Sdm car-xgb_hpolib-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d car  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-xgb_hpolib-BS+O-0 2>./stdout/err_car-xgb_hpolib-BS+O-0"
sleep 10s
screen -Sdm car-xgb_hpolib-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d car  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-xgb_hpolib-Optu-0 2>./stdout/err_car-xgb_hpolib-Optu-0"
sleep 10s
screen -Sdm car-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d car  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-xgb_blendsearch-Neve-0 2>./stdout/err_car-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm car-xgb_blendsearch-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d car  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-xgb_blendsearch-Ax-0 2>./stdout/err_car-xgb_blendsearch-Ax-0"
sleep 10s
screen -Sdm car-xgb_blendsearch-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d car  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-xgb_blendsearch-Hype-0 2>./stdout/err_car-xgb_blendsearch-Hype-0"
sleep 10s
screen -Sdm car-xgb_blendsearch-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d car  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-xgb_blendsearch-CFO-0 2>./stdout/err_car-xgb_blendsearch-CFO-0"
sleep 10s
screen -Sdm car-xgb_blendsearch-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d car  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-xgb_blendsearch-BS+O-0 2>./stdout/err_car-xgb_blendsearch-BS+O-0"
sleep 10s
screen -Sdm car-xgb_blendsearch-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d car  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-xgb_blendsearch-Optu-0 2>./stdout/err_car-xgb_blendsearch-Optu-0"
sleep 10s
screen -Sdm kc1-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d kc1  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-xgb_cfo-Neve-0 2>./stdout/err_kc1-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm kc1-xgb_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d kc1  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-xgb_cfo-Ax-0 2>./stdout/err_kc1-xgb_cfo-Ax-0"
sleep 10s
screen -Sdm kc1-xgb_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d kc1  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-xgb_cfo-Hype-0 2>./stdout/err_kc1-xgb_cfo-Hype-0"
sleep 10s
screen -Sdm kc1-xgb_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d kc1  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-xgb_cfo-CFO-0 2>./stdout/err_kc1-xgb_cfo-CFO-0"
sleep 10s
screen -Sdm kc1-xgb_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d kc1  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-xgb_cfo-BS+O-0 2>./stdout/err_kc1-xgb_cfo-BS+O-0"
sleep 10s
screen -Sdm kc1-xgb_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d kc1  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-xgb_cfo-Optu-0 2>./stdout/err_kc1-xgb_cfo-Optu-0"
sleep 10s
screen -Sdm kc1-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d kc1  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-xgb_hpolib-Neve-0 2>./stdout/err_kc1-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm kc1-xgb_hpolib-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d kc1  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-xgb_hpolib-Ax-0 2>./stdout/err_kc1-xgb_hpolib-Ax-0"
sleep 10s
screen -Sdm kc1-xgb_hpolib-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d kc1  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-xgb_hpolib-Hype-0 2>./stdout/err_kc1-xgb_hpolib-Hype-0"
sleep 10s
screen -Sdm kc1-xgb_hpolib-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d kc1  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-xgb_hpolib-CFO-0 2>./stdout/err_kc1-xgb_hpolib-CFO-0"
sleep 10s
screen -Sdm kc1-xgb_hpolib-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d kc1  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-xgb_hpolib-BS+O-0 2>./stdout/err_kc1-xgb_hpolib-BS+O-0"
sleep 10s
screen -Sdm kc1-xgb_hpolib-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d kc1  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-xgb_hpolib-Optu-0 2>./stdout/err_kc1-xgb_hpolib-Optu-0"
sleep 10s
screen -Sdm kc1-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d kc1  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-xgb_blendsearch-Neve-0 2>./stdout/err_kc1-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm kc1-xgb_blendsearch-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d kc1  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-xgb_blendsearch-Ax-0 2>./stdout/err_kc1-xgb_blendsearch-Ax-0"
sleep 10s
screen -Sdm kc1-xgb_blendsearch-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d kc1  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-xgb_blendsearch-Hype-0 2>./stdout/err_kc1-xgb_blendsearch-Hype-0"
sleep 10s
screen -Sdm kc1-xgb_blendsearch-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d kc1  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-xgb_blendsearch-CFO-0 2>./stdout/err_kc1-xgb_blendsearch-CFO-0"
sleep 10s
screen -Sdm kc1-xgb_blendsearch-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d kc1  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-xgb_blendsearch-BS+O-0 2>./stdout/err_kc1-xgb_blendsearch-BS+O-0"
sleep 10s
screen -Sdm kc1-xgb_blendsearch-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d kc1  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-xgb_blendsearch-Optu-0 2>./stdout/err_kc1-xgb_blendsearch-Optu-0"
sleep 10s
screen -Sdm kr-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d kr  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-xgb_cfo-Neve-0 2>./stdout/err_kr-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm kr-xgb_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d kr  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-xgb_cfo-Ax-0 2>./stdout/err_kr-xgb_cfo-Ax-0"
sleep 10s
screen -Sdm kr-xgb_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d kr  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-xgb_cfo-Hype-0 2>./stdout/err_kr-xgb_cfo-Hype-0"
sleep 10s
screen -Sdm kr-xgb_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d kr  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-xgb_cfo-CFO-0 2>./stdout/err_kr-xgb_cfo-CFO-0"
sleep 10s
screen -Sdm kr-xgb_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d kr  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-xgb_cfo-BS+O-0 2>./stdout/err_kr-xgb_cfo-BS+O-0"
sleep 10s
screen -Sdm kr-xgb_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d kr  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-xgb_cfo-Optu-0 2>./stdout/err_kr-xgb_cfo-Optu-0"
sleep 10s
sleep 3960s
screen -Sdm kr-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d kr  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-xgb_hpolib-Neve-0 2>./stdout/err_kr-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm kr-xgb_hpolib-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d kr  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-xgb_hpolib-Ax-0 2>./stdout/err_kr-xgb_hpolib-Ax-0"
sleep 10s
screen -Sdm kr-xgb_hpolib-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d kr  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-xgb_hpolib-Hype-0 2>./stdout/err_kr-xgb_hpolib-Hype-0"
sleep 10s
screen -Sdm kr-xgb_hpolib-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d kr  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-xgb_hpolib-CFO-0 2>./stdout/err_kr-xgb_hpolib-CFO-0"
sleep 10s
screen -Sdm kr-xgb_hpolib-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d kr  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-xgb_hpolib-BS+O-0 2>./stdout/err_kr-xgb_hpolib-BS+O-0"
sleep 10s
screen -Sdm kr-xgb_hpolib-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d kr  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-xgb_hpolib-Optu-0 2>./stdout/err_kr-xgb_hpolib-Optu-0"
sleep 10s
screen -Sdm kr-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d kr  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-xgb_blendsearch-Neve-0 2>./stdout/err_kr-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm kr-xgb_blendsearch-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d kr  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-xgb_blendsearch-Ax-0 2>./stdout/err_kr-xgb_blendsearch-Ax-0"
sleep 10s
screen -Sdm kr-xgb_blendsearch-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d kr  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-xgb_blendsearch-Hype-0 2>./stdout/err_kr-xgb_blendsearch-Hype-0"
sleep 10s
screen -Sdm kr-xgb_blendsearch-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d kr  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-xgb_blendsearch-CFO-0 2>./stdout/err_kr-xgb_blendsearch-CFO-0"
sleep 10s
screen -Sdm kr-xgb_blendsearch-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d kr  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-xgb_blendsearch-BS+O-0 2>./stdout/err_kr-xgb_blendsearch-BS+O-0"
sleep 10s
screen -Sdm kr-xgb_blendsearch-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d kr  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-xgb_blendsearch-Optu-0 2>./stdout/err_kr-xgb_blendsearch-Optu-0"
sleep 10s
screen -Sdm phon-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d phoneme  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-xgb_cfo-Neve-0 2>./stdout/err_phon-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm phon-xgb_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d phoneme  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-xgb_cfo-Ax-0 2>./stdout/err_phon-xgb_cfo-Ax-0"
sleep 10s
screen -Sdm phon-xgb_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d phoneme  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-xgb_cfo-Hype-0 2>./stdout/err_phon-xgb_cfo-Hype-0"
sleep 10s
screen -Sdm phon-xgb_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d phoneme  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-xgb_cfo-CFO-0 2>./stdout/err_phon-xgb_cfo-CFO-0"
sleep 10s
screen -Sdm phon-xgb_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d phoneme  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-xgb_cfo-BS+O-0 2>./stdout/err_phon-xgb_cfo-BS+O-0"
sleep 10s
screen -Sdm phon-xgb_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d phoneme  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-xgb_cfo-Optu-0 2>./stdout/err_phon-xgb_cfo-Optu-0"
sleep 10s
screen -Sdm phon-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d phoneme  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-xgb_hpolib-Neve-0 2>./stdout/err_phon-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm phon-xgb_hpolib-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d phoneme  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-xgb_hpolib-Ax-0 2>./stdout/err_phon-xgb_hpolib-Ax-0"
sleep 10s
screen -Sdm phon-xgb_hpolib-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d phoneme  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-xgb_hpolib-Hype-0 2>./stdout/err_phon-xgb_hpolib-Hype-0"
sleep 10s
screen -Sdm phon-xgb_hpolib-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d phoneme  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-xgb_hpolib-CFO-0 2>./stdout/err_phon-xgb_hpolib-CFO-0"
sleep 10s
screen -Sdm phon-xgb_hpolib-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d phoneme  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-xgb_hpolib-BS+O-0 2>./stdout/err_phon-xgb_hpolib-BS+O-0"
sleep 10s
screen -Sdm phon-xgb_hpolib-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d phoneme  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-xgb_hpolib-Optu-0 2>./stdout/err_phon-xgb_hpolib-Optu-0"
sleep 10s
screen -Sdm phon-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d phoneme  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-xgb_blendsearch-Neve-0 2>./stdout/err_phon-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm phon-xgb_blendsearch-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d phoneme  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-xgb_blendsearch-Ax-0 2>./stdout/err_phon-xgb_blendsearch-Ax-0"
sleep 10s
screen -Sdm phon-xgb_blendsearch-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d phoneme  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-xgb_blendsearch-Hype-0 2>./stdout/err_phon-xgb_blendsearch-Hype-0"
sleep 10s
screen -Sdm phon-xgb_blendsearch-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d phoneme  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-xgb_blendsearch-CFO-0 2>./stdout/err_phon-xgb_blendsearch-CFO-0"
sleep 10s
screen -Sdm phon-xgb_blendsearch-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d phoneme  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-xgb_blendsearch-BS+O-0 2>./stdout/err_phon-xgb_blendsearch-BS+O-0"
sleep 10s
screen -Sdm phon-xgb_blendsearch-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d phoneme  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-xgb_blendsearch-Optu-0 2>./stdout/err_phon-xgb_blendsearch-Optu-0"
sleep 10s
screen -Sdm segm-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d segment  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-xgb_cfo-Neve-0 2>./stdout/err_segm-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm segm-xgb_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d segment  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-xgb_cfo-Ax-0 2>./stdout/err_segm-xgb_cfo-Ax-0"
sleep 10s
screen -Sdm segm-xgb_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d segment  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-xgb_cfo-Hype-0 2>./stdout/err_segm-xgb_cfo-Hype-0"
sleep 10s
screen -Sdm segm-xgb_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d segment  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-xgb_cfo-CFO-0 2>./stdout/err_segm-xgb_cfo-CFO-0"
sleep 10s
screen -Sdm segm-xgb_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d segment  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-xgb_cfo-BS+O-0 2>./stdout/err_segm-xgb_cfo-BS+O-0"
sleep 10s
screen -Sdm segm-xgb_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_cfo -d segment  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-xgb_cfo-Optu-0 2>./stdout/err_segm-xgb_cfo-Optu-0"
sleep 10s
screen -Sdm segm-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d segment  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-xgb_hpolib-Neve-0 2>./stdout/err_segm-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm segm-xgb_hpolib-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d segment  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-xgb_hpolib-Ax-0 2>./stdout/err_segm-xgb_hpolib-Ax-0"
sleep 10s
screen -Sdm segm-xgb_hpolib-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d segment  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-xgb_hpolib-Hype-0 2>./stdout/err_segm-xgb_hpolib-Hype-0"
sleep 10s
screen -Sdm segm-xgb_hpolib-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d segment  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-xgb_hpolib-CFO-0 2>./stdout/err_segm-xgb_hpolib-CFO-0"
sleep 10s
screen -Sdm segm-xgb_hpolib-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d segment  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-xgb_hpolib-BS+O-0 2>./stdout/err_segm-xgb_hpolib-BS+O-0"
sleep 10s
screen -Sdm segm-xgb_hpolib-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_hpolib -d segment  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-xgb_hpolib-Optu-0 2>./stdout/err_segm-xgb_hpolib-Optu-0"
sleep 10s
screen -Sdm segm-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d segment  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-xgb_blendsearch-Neve-0 2>./stdout/err_segm-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm segm-xgb_blendsearch-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d segment  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-xgb_blendsearch-Ax-0 2>./stdout/err_segm-xgb_blendsearch-Ax-0"
sleep 10s
screen -Sdm segm-xgb_blendsearch-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d segment  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-xgb_blendsearch-Hype-0 2>./stdout/err_segm-xgb_blendsearch-Hype-0"
sleep 10s
screen -Sdm segm-xgb_blendsearch-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d segment  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-xgb_blendsearch-CFO-0 2>./stdout/err_segm-xgb_blendsearch-CFO-0"
sleep 10s
screen -Sdm segm-xgb_blendsearch-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d segment  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-xgb_blendsearch-BS+O-0 2>./stdout/err_segm-xgb_blendsearch-BS+O-0"
sleep 10s
screen -Sdm segm-xgb_blendsearch-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l xgb_blendsearch -d segment  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-xgb_blendsearch-Optu-0 2>./stdout/err_segm-xgb_blendsearch-Optu-0"
sleep 10s
sleep 3960s

screen -Sdm Airl-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d Airlines  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-xgb_cfo-Neve-0 2>./stdout/err_Airl-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm Airl-xgb_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d Airlines  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-xgb_cfo-Ax-0 2>./stdout/err_Airl-xgb_cfo-Ax-0"
sleep 10s
screen -Sdm Airl-xgb_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d Airlines  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-xgb_cfo-Hype-0 2>./stdout/err_Airl-xgb_cfo-Hype-0"
sleep 10s
screen -Sdm Airl-xgb_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d Airlines  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-xgb_cfo-CFO-0 2>./stdout/err_Airl-xgb_cfo-CFO-0"
sleep 10s
screen -Sdm Airl-xgb_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d Airlines  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-xgb_cfo-BS+O-0 2>./stdout/err_Airl-xgb_cfo-BS+O-0"
sleep 10s
screen -Sdm Airl-xgb_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d Airlines  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-xgb_cfo-Optu-0 2>./stdout/err_Airl-xgb_cfo-Optu-0"
sleep 10s
screen -Sdm Airl-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d Airlines  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-xgb_hpolib-Neve-0 2>./stdout/err_Airl-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm Airl-xgb_hpolib-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d Airlines  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-xgb_hpolib-Ax-0 2>./stdout/err_Airl-xgb_hpolib-Ax-0"
sleep 10s
screen -Sdm Airl-xgb_hpolib-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d Airlines  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-xgb_hpolib-Hype-0 2>./stdout/err_Airl-xgb_hpolib-Hype-0"
sleep 10s
screen -Sdm Airl-xgb_hpolib-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d Airlines  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-xgb_hpolib-CFO-0 2>./stdout/err_Airl-xgb_hpolib-CFO-0"
sleep 10s
screen -Sdm Airl-xgb_hpolib-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d Airlines  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-xgb_hpolib-BS+O-0 2>./stdout/err_Airl-xgb_hpolib-BS+O-0"
sleep 10s
screen -Sdm Airl-xgb_hpolib-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d Airlines  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-xgb_hpolib-Optu-0 2>./stdout/err_Airl-xgb_hpolib-Optu-0"
sleep 10s
screen -Sdm Airl-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d Airlines  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-xgb_blendsearch-Neve-0 2>./stdout/err_Airl-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm Airl-xgb_blendsearch-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d Airlines  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-xgb_blendsearch-Ax-0 2>./stdout/err_Airl-xgb_blendsearch-Ax-0"
sleep 10s
screen -Sdm Airl-xgb_blendsearch-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d Airlines  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-xgb_blendsearch-Hype-0 2>./stdout/err_Airl-xgb_blendsearch-Hype-0"
sleep 10s
screen -Sdm Airl-xgb_blendsearch-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d Airlines  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-xgb_blendsearch-CFO-0 2>./stdout/err_Airl-xgb_blendsearch-CFO-0"
sleep 10s
screen -Sdm Airl-xgb_blendsearch-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d Airlines  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-xgb_blendsearch-BS+O-0 2>./stdout/err_Airl-xgb_blendsearch-BS+O-0"
sleep 10s
screen -Sdm Airl-xgb_blendsearch-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d Airlines  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-xgb_blendsearch-Optu-0 2>./stdout/err_Airl-xgb_blendsearch-Optu-0"
sleep 10s
screen -Sdm chri-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d christine  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-xgb_cfo-Neve-0 2>./stdout/err_chri-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm chri-xgb_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d christine  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-xgb_cfo-Ax-0 2>./stdout/err_chri-xgb_cfo-Ax-0"
sleep 10s
screen -Sdm chri-xgb_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d christine  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-xgb_cfo-Hype-0 2>./stdout/err_chri-xgb_cfo-Hype-0"
sleep 10s
screen -Sdm chri-xgb_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d christine  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-xgb_cfo-CFO-0 2>./stdout/err_chri-xgb_cfo-CFO-0"
sleep 10s
screen -Sdm chri-xgb_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d christine  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-xgb_cfo-BS+O-0 2>./stdout/err_chri-xgb_cfo-BS+O-0"
sleep 10s
screen -Sdm chri-xgb_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d christine  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-xgb_cfo-Optu-0 2>./stdout/err_chri-xgb_cfo-Optu-0"
sleep 10s
screen -Sdm chri-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d christine  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-xgb_hpolib-Neve-0 2>./stdout/err_chri-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm chri-xgb_hpolib-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d christine  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-xgb_hpolib-Ax-0 2>./stdout/err_chri-xgb_hpolib-Ax-0"
sleep 10s
screen -Sdm chri-xgb_hpolib-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d christine  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-xgb_hpolib-Hype-0 2>./stdout/err_chri-xgb_hpolib-Hype-0"
sleep 10s
screen -Sdm chri-xgb_hpolib-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d christine  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-xgb_hpolib-CFO-0 2>./stdout/err_chri-xgb_hpolib-CFO-0"
sleep 10s
screen -Sdm chri-xgb_hpolib-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d christine  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-xgb_hpolib-BS+O-0 2>./stdout/err_chri-xgb_hpolib-BS+O-0"
sleep 10s
screen -Sdm chri-xgb_hpolib-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d christine  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-xgb_hpolib-Optu-0 2>./stdout/err_chri-xgb_hpolib-Optu-0"
sleep 10s
screen -Sdm chri-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d christine  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-xgb_blendsearch-Neve-0 2>./stdout/err_chri-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm chri-xgb_blendsearch-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d christine  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-xgb_blendsearch-Ax-0 2>./stdout/err_chri-xgb_blendsearch-Ax-0"
sleep 10s
screen -Sdm chri-xgb_blendsearch-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d christine  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-xgb_blendsearch-Hype-0 2>./stdout/err_chri-xgb_blendsearch-Hype-0"
sleep 10s
screen -Sdm chri-xgb_blendsearch-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d christine  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-xgb_blendsearch-CFO-0 2>./stdout/err_chri-xgb_blendsearch-CFO-0"
sleep 10s
screen -Sdm chri-xgb_blendsearch-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d christine  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-xgb_blendsearch-BS+O-0 2>./stdout/err_chri-xgb_blendsearch-BS+O-0"
sleep 10s
screen -Sdm chri-xgb_blendsearch-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d christine  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-xgb_blendsearch-Optu-0 2>./stdout/err_chri-xgb_blendsearch-Optu-0"
sleep 10s
screen -Sdm shut-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d shuttle  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-xgb_cfo-Neve-0 2>./stdout/err_shut-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm shut-xgb_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d shuttle  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-xgb_cfo-Ax-0 2>./stdout/err_shut-xgb_cfo-Ax-0"
sleep 10s
screen -Sdm shut-xgb_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d shuttle  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-xgb_cfo-Hype-0 2>./stdout/err_shut-xgb_cfo-Hype-0"
sleep 10s
screen -Sdm shut-xgb_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d shuttle  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-xgb_cfo-CFO-0 2>./stdout/err_shut-xgb_cfo-CFO-0"
sleep 10s
screen -Sdm shut-xgb_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d shuttle  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-xgb_cfo-BS+O-0 2>./stdout/err_shut-xgb_cfo-BS+O-0"
sleep 10s
screen -Sdm shut-xgb_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d shuttle  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-xgb_cfo-Optu-0 2>./stdout/err_shut-xgb_cfo-Optu-0"
sleep 10s
screen -Sdm shut-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d shuttle  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-xgb_hpolib-Neve-0 2>./stdout/err_shut-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm shut-xgb_hpolib-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d shuttle  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-xgb_hpolib-Ax-0 2>./stdout/err_shut-xgb_hpolib-Ax-0"
sleep 10s
screen -Sdm shut-xgb_hpolib-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d shuttle  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-xgb_hpolib-Hype-0 2>./stdout/err_shut-xgb_hpolib-Hype-0"
sleep 10s
screen -Sdm shut-xgb_hpolib-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d shuttle  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-xgb_hpolib-CFO-0 2>./stdout/err_shut-xgb_hpolib-CFO-0"
sleep 10s
screen -Sdm shut-xgb_hpolib-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d shuttle  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-xgb_hpolib-BS+O-0 2>./stdout/err_shut-xgb_hpolib-BS+O-0"
sleep 10s
screen -Sdm shut-xgb_hpolib-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d shuttle  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-xgb_hpolib-Optu-0 2>./stdout/err_shut-xgb_hpolib-Optu-0"
sleep 10s
sleep 15840s
screen -Sdm shut-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d shuttle  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-xgb_blendsearch-Neve-0 2>./stdout/err_shut-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm shut-xgb_blendsearch-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d shuttle  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-xgb_blendsearch-Ax-0 2>./stdout/err_shut-xgb_blendsearch-Ax-0"
sleep 10s
screen -Sdm shut-xgb_blendsearch-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d shuttle  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-xgb_blendsearch-Hype-0 2>./stdout/err_shut-xgb_blendsearch-Hype-0"
sleep 10s
screen -Sdm shut-xgb_blendsearch-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d shuttle  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-xgb_blendsearch-CFO-0 2>./stdout/err_shut-xgb_blendsearch-CFO-0"
sleep 10s
screen -Sdm shut-xgb_blendsearch-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d shuttle  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-xgb_blendsearch-BS+O-0 2>./stdout/err_shut-xgb_blendsearch-BS+O-0"
sleep 10s
screen -Sdm shut-xgb_blendsearch-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d shuttle  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-xgb_blendsearch-Optu-0 2>./stdout/err_shut-xgb_blendsearch-Optu-0"
sleep 10s
screen -Sdm conn-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d connect  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-xgb_cfo-Neve-0 2>./stdout/err_conn-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm conn-xgb_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d connect  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-xgb_cfo-Ax-0 2>./stdout/err_conn-xgb_cfo-Ax-0"
sleep 10s
screen -Sdm conn-xgb_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d connect  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-xgb_cfo-Hype-0 2>./stdout/err_conn-xgb_cfo-Hype-0"
sleep 10s
screen -Sdm conn-xgb_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d connect  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-xgb_cfo-CFO-0 2>./stdout/err_conn-xgb_cfo-CFO-0"
sleep 10s
screen -Sdm conn-xgb_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d connect  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-xgb_cfo-BS+O-0 2>./stdout/err_conn-xgb_cfo-BS+O-0"
sleep 10s
screen -Sdm conn-xgb_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d connect  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-xgb_cfo-Optu-0 2>./stdout/err_conn-xgb_cfo-Optu-0"
sleep 10s
screen -Sdm conn-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d connect  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-xgb_hpolib-Neve-0 2>./stdout/err_conn-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm conn-xgb_hpolib-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d connect  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-xgb_hpolib-Ax-0 2>./stdout/err_conn-xgb_hpolib-Ax-0"
sleep 10s
screen -Sdm conn-xgb_hpolib-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d connect  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-xgb_hpolib-Hype-0 2>./stdout/err_conn-xgb_hpolib-Hype-0"
sleep 10s
screen -Sdm conn-xgb_hpolib-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d connect  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-xgb_hpolib-CFO-0 2>./stdout/err_conn-xgb_hpolib-CFO-0"
sleep 10s
screen -Sdm conn-xgb_hpolib-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d connect  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-xgb_hpolib-BS+O-0 2>./stdout/err_conn-xgb_hpolib-BS+O-0"
sleep 10s
screen -Sdm conn-xgb_hpolib-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d connect  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-xgb_hpolib-Optu-0 2>./stdout/err_conn-xgb_hpolib-Optu-0"
sleep 10s
screen -Sdm conn-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d connect  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-xgb_blendsearch-Neve-0 2>./stdout/err_conn-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm conn-xgb_blendsearch-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d connect  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-xgb_blendsearch-Ax-0 2>./stdout/err_conn-xgb_blendsearch-Ax-0"
sleep 10s
screen -Sdm conn-xgb_blendsearch-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d connect  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-xgb_blendsearch-Hype-0 2>./stdout/err_conn-xgb_blendsearch-Hype-0"
sleep 10s
screen -Sdm conn-xgb_blendsearch-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d connect  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-xgb_blendsearch-CFO-0 2>./stdout/err_conn-xgb_blendsearch-CFO-0"
sleep 10s
screen -Sdm conn-xgb_blendsearch-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d connect  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-xgb_blendsearch-BS+O-0 2>./stdout/err_conn-xgb_blendsearch-BS+O-0"
sleep 10s
screen -Sdm conn-xgb_blendsearch-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d connect  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-xgb_blendsearch-Optu-0 2>./stdout/err_conn-xgb_blendsearch-Optu-0"
sleep 10s
screen -Sdm sylv-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d sylvine  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-xgb_cfo-Neve-0 2>./stdout/err_sylv-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm sylv-xgb_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d sylvine  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-xgb_cfo-Ax-0 2>./stdout/err_sylv-xgb_cfo-Ax-0"
sleep 10s
screen -Sdm sylv-xgb_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d sylvine  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-xgb_cfo-Hype-0 2>./stdout/err_sylv-xgb_cfo-Hype-0"
sleep 10s
screen -Sdm sylv-xgb_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d sylvine  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-xgb_cfo-CFO-0 2>./stdout/err_sylv-xgb_cfo-CFO-0"
sleep 10s
screen -Sdm sylv-xgb_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d sylvine  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-xgb_cfo-BS+O-0 2>./stdout/err_sylv-xgb_cfo-BS+O-0"
sleep 10s
screen -Sdm sylv-xgb_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d sylvine  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-xgb_cfo-Optu-0 2>./stdout/err_sylv-xgb_cfo-Optu-0"
sleep 10s
screen -Sdm sylv-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d sylvine  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-xgb_hpolib-Neve-0 2>./stdout/err_sylv-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm sylv-xgb_hpolib-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d sylvine  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-xgb_hpolib-Ax-0 2>./stdout/err_sylv-xgb_hpolib-Ax-0"
sleep 10s
screen -Sdm sylv-xgb_hpolib-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d sylvine  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-xgb_hpolib-Hype-0 2>./stdout/err_sylv-xgb_hpolib-Hype-0"
sleep 10s
screen -Sdm sylv-xgb_hpolib-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d sylvine  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-xgb_hpolib-CFO-0 2>./stdout/err_sylv-xgb_hpolib-CFO-0"
sleep 10s
screen -Sdm sylv-xgb_hpolib-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d sylvine  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-xgb_hpolib-BS+O-0 2>./stdout/err_sylv-xgb_hpolib-BS+O-0"
sleep 10s
screen -Sdm sylv-xgb_hpolib-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d sylvine  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-xgb_hpolib-Optu-0 2>./stdout/err_sylv-xgb_hpolib-Optu-0"
sleep 10s
screen -Sdm sylv-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d sylvine  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-xgb_blendsearch-Neve-0 2>./stdout/err_sylv-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm sylv-xgb_blendsearch-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d sylvine  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-xgb_blendsearch-Ax-0 2>./stdout/err_sylv-xgb_blendsearch-Ax-0"
sleep 10s
screen -Sdm sylv-xgb_blendsearch-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d sylvine  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-xgb_blendsearch-Hype-0 2>./stdout/err_sylv-xgb_blendsearch-Hype-0"
sleep 10s
screen -Sdm sylv-xgb_blendsearch-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d sylvine  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-xgb_blendsearch-CFO-0 2>./stdout/err_sylv-xgb_blendsearch-CFO-0"
sleep 10s
screen -Sdm sylv-xgb_blendsearch-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d sylvine  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-xgb_blendsearch-BS+O-0 2>./stdout/err_sylv-xgb_blendsearch-BS+O-0"
sleep 10s
screen -Sdm sylv-xgb_blendsearch-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d sylvine  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-xgb_blendsearch-Optu-0 2>./stdout/err_sylv-xgb_blendsearch-Optu-0"
sleep 10s
screen -Sdm guil-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d guillermo  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-xgb_cfo-Neve-0 2>./stdout/err_guil-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm guil-xgb_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d guillermo  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-xgb_cfo-Ax-0 2>./stdout/err_guil-xgb_cfo-Ax-0"
sleep 10s
screen -Sdm guil-xgb_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d guillermo  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-xgb_cfo-Hype-0 2>./stdout/err_guil-xgb_cfo-Hype-0"
sleep 10s
screen -Sdm guil-xgb_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d guillermo  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-xgb_cfo-CFO-0 2>./stdout/err_guil-xgb_cfo-CFO-0"
sleep 10s
screen -Sdm guil-xgb_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d guillermo  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-xgb_cfo-BS+O-0 2>./stdout/err_guil-xgb_cfo-BS+O-0"
sleep 10s
screen -Sdm guil-xgb_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d guillermo  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-xgb_cfo-Optu-0 2>./stdout/err_guil-xgb_cfo-Optu-0"
sleep 10s
sleep 15840s
screen -Sdm guil-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d guillermo  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-xgb_hpolib-Neve-0 2>./stdout/err_guil-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm guil-xgb_hpolib-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d guillermo  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-xgb_hpolib-Ax-0 2>./stdout/err_guil-xgb_hpolib-Ax-0"
sleep 10s
screen -Sdm guil-xgb_hpolib-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d guillermo  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-xgb_hpolib-Hype-0 2>./stdout/err_guil-xgb_hpolib-Hype-0"
sleep 10s
screen -Sdm guil-xgb_hpolib-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d guillermo  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-xgb_hpolib-CFO-0 2>./stdout/err_guil-xgb_hpolib-CFO-0"
sleep 10s
screen -Sdm guil-xgb_hpolib-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d guillermo  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-xgb_hpolib-BS+O-0 2>./stdout/err_guil-xgb_hpolib-BS+O-0"
sleep 10s
screen -Sdm guil-xgb_hpolib-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d guillermo  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-xgb_hpolib-Optu-0 2>./stdout/err_guil-xgb_hpolib-Optu-0"
sleep 10s
screen -Sdm guil-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d guillermo  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-xgb_blendsearch-Neve-0 2>./stdout/err_guil-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm guil-xgb_blendsearch-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d guillermo  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-xgb_blendsearch-Ax-0 2>./stdout/err_guil-xgb_blendsearch-Ax-0"
sleep 10s
screen -Sdm guil-xgb_blendsearch-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d guillermo  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-xgb_blendsearch-Hype-0 2>./stdout/err_guil-xgb_blendsearch-Hype-0"
sleep 10s
screen -Sdm guil-xgb_blendsearch-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d guillermo  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-xgb_blendsearch-CFO-0 2>./stdout/err_guil-xgb_blendsearch-CFO-0"
sleep 10s
screen -Sdm guil-xgb_blendsearch-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d guillermo  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-xgb_blendsearch-BS+O-0 2>./stdout/err_guil-xgb_blendsearch-BS+O-0"
sleep 10s
screen -Sdm guil-xgb_blendsearch-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d guillermo  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-xgb_blendsearch-Optu-0 2>./stdout/err_guil-xgb_blendsearch-Optu-0"
sleep 10s
screen -Sdm volk-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d volkert  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-xgb_cfo-Neve-0 2>./stdout/err_volk-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm volk-xgb_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d volkert  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-xgb_cfo-Ax-0 2>./stdout/err_volk-xgb_cfo-Ax-0"
sleep 10s
screen -Sdm volk-xgb_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d volkert  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-xgb_cfo-Hype-0 2>./stdout/err_volk-xgb_cfo-Hype-0"
sleep 10s
screen -Sdm volk-xgb_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d volkert  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-xgb_cfo-CFO-0 2>./stdout/err_volk-xgb_cfo-CFO-0"
sleep 10s
screen -Sdm volk-xgb_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d volkert  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-xgb_cfo-BS+O-0 2>./stdout/err_volk-xgb_cfo-BS+O-0"
sleep 10s
screen -Sdm volk-xgb_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d volkert  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-xgb_cfo-Optu-0 2>./stdout/err_volk-xgb_cfo-Optu-0"
sleep 10s
screen -Sdm volk-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d volkert  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-xgb_hpolib-Neve-0 2>./stdout/err_volk-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm volk-xgb_hpolib-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d volkert  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-xgb_hpolib-Ax-0 2>./stdout/err_volk-xgb_hpolib-Ax-0"
sleep 10s
screen -Sdm volk-xgb_hpolib-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d volkert  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-xgb_hpolib-Hype-0 2>./stdout/err_volk-xgb_hpolib-Hype-0"
sleep 10s
screen -Sdm volk-xgb_hpolib-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d volkert  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-xgb_hpolib-CFO-0 2>./stdout/err_volk-xgb_hpolib-CFO-0"
sleep 10s
screen -Sdm volk-xgb_hpolib-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d volkert  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-xgb_hpolib-BS+O-0 2>./stdout/err_volk-xgb_hpolib-BS+O-0"
sleep 10s
screen -Sdm volk-xgb_hpolib-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d volkert  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-xgb_hpolib-Optu-0 2>./stdout/err_volk-xgb_hpolib-Optu-0"
sleep 10s
screen -Sdm volk-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d volkert  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-xgb_blendsearch-Neve-0 2>./stdout/err_volk-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm volk-xgb_blendsearch-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d volkert  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-xgb_blendsearch-Ax-0 2>./stdout/err_volk-xgb_blendsearch-Ax-0"
sleep 10s
screen -Sdm volk-xgb_blendsearch-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d volkert  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-xgb_blendsearch-Hype-0 2>./stdout/err_volk-xgb_blendsearch-Hype-0"
sleep 10s
screen -Sdm volk-xgb_blendsearch-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d volkert  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-xgb_blendsearch-CFO-0 2>./stdout/err_volk-xgb_blendsearch-CFO-0"
sleep 10s
screen -Sdm volk-xgb_blendsearch-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d volkert  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-xgb_blendsearch-BS+O-0 2>./stdout/err_volk-xgb_blendsearch-BS+O-0"
sleep 10s
screen -Sdm volk-xgb_blendsearch-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d volkert  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-xgb_blendsearch-Optu-0 2>./stdout/err_volk-xgb_blendsearch-Optu-0"
sleep 10s
screen -Sdm Mini-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d MiniBooNE  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-xgb_cfo-Neve-0 2>./stdout/err_Mini-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm Mini-xgb_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d MiniBooNE  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-xgb_cfo-Ax-0 2>./stdout/err_Mini-xgb_cfo-Ax-0"
sleep 10s
screen -Sdm Mini-xgb_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d MiniBooNE  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-xgb_cfo-Hype-0 2>./stdout/err_Mini-xgb_cfo-Hype-0"
sleep 10s
screen -Sdm Mini-xgb_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d MiniBooNE  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-xgb_cfo-CFO-0 2>./stdout/err_Mini-xgb_cfo-CFO-0"
sleep 10s
screen -Sdm Mini-xgb_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d MiniBooNE  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-xgb_cfo-BS+O-0 2>./stdout/err_Mini-xgb_cfo-BS+O-0"
sleep 10s
screen -Sdm Mini-xgb_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d MiniBooNE  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-xgb_cfo-Optu-0 2>./stdout/err_Mini-xgb_cfo-Optu-0"
sleep 10s
screen -Sdm Mini-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d MiniBooNE  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-xgb_hpolib-Neve-0 2>./stdout/err_Mini-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm Mini-xgb_hpolib-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d MiniBooNE  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-xgb_hpolib-Ax-0 2>./stdout/err_Mini-xgb_hpolib-Ax-0"
sleep 10s
screen -Sdm Mini-xgb_hpolib-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d MiniBooNE  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-xgb_hpolib-Hype-0 2>./stdout/err_Mini-xgb_hpolib-Hype-0"
sleep 10s
screen -Sdm Mini-xgb_hpolib-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d MiniBooNE  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-xgb_hpolib-CFO-0 2>./stdout/err_Mini-xgb_hpolib-CFO-0"
sleep 10s
screen -Sdm Mini-xgb_hpolib-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d MiniBooNE  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-xgb_hpolib-BS+O-0 2>./stdout/err_Mini-xgb_hpolib-BS+O-0"
sleep 10s
screen -Sdm Mini-xgb_hpolib-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d MiniBooNE  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-xgb_hpolib-Optu-0 2>./stdout/err_Mini-xgb_hpolib-Optu-0"
sleep 10s
screen -Sdm Mini-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d MiniBooNE  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-xgb_blendsearch-Neve-0 2>./stdout/err_Mini-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm Mini-xgb_blendsearch-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d MiniBooNE  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-xgb_blendsearch-Ax-0 2>./stdout/err_Mini-xgb_blendsearch-Ax-0"
sleep 10s
screen -Sdm Mini-xgb_blendsearch-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d MiniBooNE  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-xgb_blendsearch-Hype-0 2>./stdout/err_Mini-xgb_blendsearch-Hype-0"
sleep 10s
screen -Sdm Mini-xgb_blendsearch-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d MiniBooNE  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-xgb_blendsearch-CFO-0 2>./stdout/err_Mini-xgb_blendsearch-CFO-0"
sleep 10s
screen -Sdm Mini-xgb_blendsearch-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d MiniBooNE  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-xgb_blendsearch-BS+O-0 2>./stdout/err_Mini-xgb_blendsearch-BS+O-0"
sleep 10s
screen -Sdm Mini-xgb_blendsearch-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d MiniBooNE  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-xgb_blendsearch-Optu-0 2>./stdout/err_Mini-xgb_blendsearch-Optu-0"
sleep 10s
sleep 15840s
screen -Sdm Jann-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d Jannis  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-xgb_cfo-Neve-0 2>./stdout/err_Jann-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm Jann-xgb_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d Jannis  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-xgb_cfo-Ax-0 2>./stdout/err_Jann-xgb_cfo-Ax-0"
sleep 10s
screen -Sdm Jann-xgb_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d Jannis  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-xgb_cfo-Hype-0 2>./stdout/err_Jann-xgb_cfo-Hype-0"
sleep 10s
screen -Sdm Jann-xgb_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d Jannis  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-xgb_cfo-CFO-0 2>./stdout/err_Jann-xgb_cfo-CFO-0"
sleep 10s
screen -Sdm Jann-xgb_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d Jannis  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-xgb_cfo-BS+O-0 2>./stdout/err_Jann-xgb_cfo-BS+O-0"
sleep 10s
screen -Sdm Jann-xgb_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d Jannis  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-xgb_cfo-Optu-0 2>./stdout/err_Jann-xgb_cfo-Optu-0"
sleep 10s
screen -Sdm Jann-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d Jannis  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-xgb_hpolib-Neve-0 2>./stdout/err_Jann-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm Jann-xgb_hpolib-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d Jannis  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-xgb_hpolib-Ax-0 2>./stdout/err_Jann-xgb_hpolib-Ax-0"
sleep 10s
screen -Sdm Jann-xgb_hpolib-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d Jannis  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-xgb_hpolib-Hype-0 2>./stdout/err_Jann-xgb_hpolib-Hype-0"
sleep 10s
screen -Sdm Jann-xgb_hpolib-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d Jannis  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-xgb_hpolib-CFO-0 2>./stdout/err_Jann-xgb_hpolib-CFO-0"
sleep 10s
screen -Sdm Jann-xgb_hpolib-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d Jannis  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-xgb_hpolib-BS+O-0 2>./stdout/err_Jann-xgb_hpolib-BS+O-0"
sleep 10s
screen -Sdm Jann-xgb_hpolib-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d Jannis  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-xgb_hpolib-Optu-0 2>./stdout/err_Jann-xgb_hpolib-Optu-0"
sleep 10s
screen -Sdm Jann-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d Jannis  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-xgb_blendsearch-Neve-0 2>./stdout/err_Jann-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm Jann-xgb_blendsearch-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d Jannis  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-xgb_blendsearch-Ax-0 2>./stdout/err_Jann-xgb_blendsearch-Ax-0"
sleep 10s
screen -Sdm Jann-xgb_blendsearch-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d Jannis  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-xgb_blendsearch-Hype-0 2>./stdout/err_Jann-xgb_blendsearch-Hype-0"
sleep 10s
screen -Sdm Jann-xgb_blendsearch-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d Jannis  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-xgb_blendsearch-CFO-0 2>./stdout/err_Jann-xgb_blendsearch-CFO-0"
sleep 10s
screen -Sdm Jann-xgb_blendsearch-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d Jannis  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-xgb_blendsearch-BS+O-0 2>./stdout/err_Jann-xgb_blendsearch-BS+O-0"
sleep 10s
screen -Sdm Jann-xgb_blendsearch-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d Jannis  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-xgb_blendsearch-Optu-0 2>./stdout/err_Jann-xgb_blendsearch-Optu-0"
sleep 10s
screen -Sdm mfea-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d mfeat  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-xgb_cfo-Neve-0 2>./stdout/err_mfea-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm mfea-xgb_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d mfeat  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-xgb_cfo-Ax-0 2>./stdout/err_mfea-xgb_cfo-Ax-0"
sleep 10s
screen -Sdm mfea-xgb_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d mfeat  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-xgb_cfo-Hype-0 2>./stdout/err_mfea-xgb_cfo-Hype-0"
sleep 10s
screen -Sdm mfea-xgb_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d mfeat  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-xgb_cfo-CFO-0 2>./stdout/err_mfea-xgb_cfo-CFO-0"
sleep 10s
screen -Sdm mfea-xgb_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d mfeat  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-xgb_cfo-BS+O-0 2>./stdout/err_mfea-xgb_cfo-BS+O-0"
sleep 10s
screen -Sdm mfea-xgb_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d mfeat  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-xgb_cfo-Optu-0 2>./stdout/err_mfea-xgb_cfo-Optu-0"
sleep 10s
screen -Sdm mfea-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d mfeat  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-xgb_hpolib-Neve-0 2>./stdout/err_mfea-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm mfea-xgb_hpolib-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d mfeat  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-xgb_hpolib-Ax-0 2>./stdout/err_mfea-xgb_hpolib-Ax-0"
sleep 10s
screen -Sdm mfea-xgb_hpolib-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d mfeat  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-xgb_hpolib-Hype-0 2>./stdout/err_mfea-xgb_hpolib-Hype-0"
sleep 10s
screen -Sdm mfea-xgb_hpolib-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d mfeat  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-xgb_hpolib-CFO-0 2>./stdout/err_mfea-xgb_hpolib-CFO-0"
sleep 10s
screen -Sdm mfea-xgb_hpolib-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d mfeat  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-xgb_hpolib-BS+O-0 2>./stdout/err_mfea-xgb_hpolib-BS+O-0"
sleep 10s
screen -Sdm mfea-xgb_hpolib-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d mfeat  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-xgb_hpolib-Optu-0 2>./stdout/err_mfea-xgb_hpolib-Optu-0"
sleep 10s
screen -Sdm mfea-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d mfeat  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-xgb_blendsearch-Neve-0 2>./stdout/err_mfea-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm mfea-xgb_blendsearch-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d mfeat  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-xgb_blendsearch-Ax-0 2>./stdout/err_mfea-xgb_blendsearch-Ax-0"
sleep 10s
screen -Sdm mfea-xgb_blendsearch-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d mfeat  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-xgb_blendsearch-Hype-0 2>./stdout/err_mfea-xgb_blendsearch-Hype-0"
sleep 10s
screen -Sdm mfea-xgb_blendsearch-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d mfeat  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-xgb_blendsearch-CFO-0 2>./stdout/err_mfea-xgb_blendsearch-CFO-0"
sleep 10s
screen -Sdm mfea-xgb_blendsearch-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d mfeat  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-xgb_blendsearch-BS+O-0 2>./stdout/err_mfea-xgb_blendsearch-BS+O-0"
sleep 10s
screen -Sdm mfea-xgb_blendsearch-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d mfeat  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-xgb_blendsearch-Optu-0 2>./stdout/err_mfea-xgb_blendsearch-Optu-0"
sleep 10s
screen -Sdm jung-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d jungle  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-xgb_cfo-Neve-0 2>./stdout/err_jung-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm jung-xgb_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d jungle  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-xgb_cfo-Ax-0 2>./stdout/err_jung-xgb_cfo-Ax-0"
sleep 10s
screen -Sdm jung-xgb_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d jungle  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-xgb_cfo-Hype-0 2>./stdout/err_jung-xgb_cfo-Hype-0"
sleep 10s
screen -Sdm jung-xgb_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d jungle  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-xgb_cfo-CFO-0 2>./stdout/err_jung-xgb_cfo-CFO-0"
sleep 10s
screen -Sdm jung-xgb_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d jungle  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-xgb_cfo-BS+O-0 2>./stdout/err_jung-xgb_cfo-BS+O-0"
sleep 10s
screen -Sdm jung-xgb_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d jungle  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-xgb_cfo-Optu-0 2>./stdout/err_jung-xgb_cfo-Optu-0"
sleep 10s
screen -Sdm jung-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d jungle  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-xgb_hpolib-Neve-0 2>./stdout/err_jung-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm jung-xgb_hpolib-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d jungle  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-xgb_hpolib-Ax-0 2>./stdout/err_jung-xgb_hpolib-Ax-0"
sleep 10s
screen -Sdm jung-xgb_hpolib-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d jungle  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-xgb_hpolib-Hype-0 2>./stdout/err_jung-xgb_hpolib-Hype-0"
sleep 10s
screen -Sdm jung-xgb_hpolib-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d jungle  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-xgb_hpolib-CFO-0 2>./stdout/err_jung-xgb_hpolib-CFO-0"
sleep 10s
screen -Sdm jung-xgb_hpolib-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d jungle  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-xgb_hpolib-BS+O-0 2>./stdout/err_jung-xgb_hpolib-BS+O-0"
sleep 10s
screen -Sdm jung-xgb_hpolib-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d jungle  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-xgb_hpolib-Optu-0 2>./stdout/err_jung-xgb_hpolib-Optu-0"
sleep 10s
sleep 15840s
screen -Sdm jung-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d jungle  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-xgb_blendsearch-Neve-0 2>./stdout/err_jung-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm jung-xgb_blendsearch-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d jungle  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-xgb_blendsearch-Ax-0 2>./stdout/err_jung-xgb_blendsearch-Ax-0"
sleep 10s
screen -Sdm jung-xgb_blendsearch-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d jungle  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-xgb_blendsearch-Hype-0 2>./stdout/err_jung-xgb_blendsearch-Hype-0"
sleep 10s
screen -Sdm jung-xgb_blendsearch-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d jungle  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-xgb_blendsearch-CFO-0 2>./stdout/err_jung-xgb_blendsearch-CFO-0"
sleep 10s
screen -Sdm jung-xgb_blendsearch-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d jungle  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-xgb_blendsearch-BS+O-0 2>./stdout/err_jung-xgb_blendsearch-BS+O-0"
sleep 10s
screen -Sdm jung-xgb_blendsearch-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d jungle  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-xgb_blendsearch-Optu-0 2>./stdout/err_jung-xgb_blendsearch-Optu-0"
sleep 10s
screen -Sdm jasm-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d jasmine  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-xgb_cfo-Neve-0 2>./stdout/err_jasm-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm jasm-xgb_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d jasmine  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-xgb_cfo-Ax-0 2>./stdout/err_jasm-xgb_cfo-Ax-0"
sleep 10s
screen -Sdm jasm-xgb_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d jasmine  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-xgb_cfo-Hype-0 2>./stdout/err_jasm-xgb_cfo-Hype-0"
sleep 10s
screen -Sdm jasm-xgb_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d jasmine  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-xgb_cfo-CFO-0 2>./stdout/err_jasm-xgb_cfo-CFO-0"
sleep 10s
screen -Sdm jasm-xgb_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d jasmine  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-xgb_cfo-BS+O-0 2>./stdout/err_jasm-xgb_cfo-BS+O-0"
sleep 10s
screen -Sdm jasm-xgb_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d jasmine  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-xgb_cfo-Optu-0 2>./stdout/err_jasm-xgb_cfo-Optu-0"
sleep 10s
screen -Sdm jasm-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d jasmine  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-xgb_hpolib-Neve-0 2>./stdout/err_jasm-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm jasm-xgb_hpolib-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d jasmine  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-xgb_hpolib-Ax-0 2>./stdout/err_jasm-xgb_hpolib-Ax-0"
sleep 10s
screen -Sdm jasm-xgb_hpolib-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d jasmine  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-xgb_hpolib-Hype-0 2>./stdout/err_jasm-xgb_hpolib-Hype-0"
sleep 10s
screen -Sdm jasm-xgb_hpolib-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d jasmine  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-xgb_hpolib-CFO-0 2>./stdout/err_jasm-xgb_hpolib-CFO-0"
sleep 10s
screen -Sdm jasm-xgb_hpolib-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d jasmine  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-xgb_hpolib-BS+O-0 2>./stdout/err_jasm-xgb_hpolib-BS+O-0"
sleep 10s
screen -Sdm jasm-xgb_hpolib-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d jasmine  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-xgb_hpolib-Optu-0 2>./stdout/err_jasm-xgb_hpolib-Optu-0"
sleep 10s
screen -Sdm jasm-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d jasmine  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-xgb_blendsearch-Neve-0 2>./stdout/err_jasm-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm jasm-xgb_blendsearch-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d jasmine  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-xgb_blendsearch-Ax-0 2>./stdout/err_jasm-xgb_blendsearch-Ax-0"
sleep 10s
screen -Sdm jasm-xgb_blendsearch-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d jasmine  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-xgb_blendsearch-Hype-0 2>./stdout/err_jasm-xgb_blendsearch-Hype-0"
sleep 10s
screen -Sdm jasm-xgb_blendsearch-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d jasmine  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-xgb_blendsearch-CFO-0 2>./stdout/err_jasm-xgb_blendsearch-CFO-0"
sleep 10s
screen -Sdm jasm-xgb_blendsearch-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d jasmine  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-xgb_blendsearch-BS+O-0 2>./stdout/err_jasm-xgb_blendsearch-BS+O-0"
sleep 10s
screen -Sdm jasm-xgb_blendsearch-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d jasmine  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-xgb_blendsearch-Optu-0 2>./stdout/err_jasm-xgb_blendsearch-Optu-0"
sleep 10s
screen -Sdm ricc-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d riccardo  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-xgb_cfo-Neve-0 2>./stdout/err_ricc-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm ricc-xgb_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d riccardo  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-xgb_cfo-Ax-0 2>./stdout/err_ricc-xgb_cfo-Ax-0"
sleep 10s
screen -Sdm ricc-xgb_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d riccardo  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-xgb_cfo-Hype-0 2>./stdout/err_ricc-xgb_cfo-Hype-0"
sleep 10s
screen -Sdm ricc-xgb_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d riccardo  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-xgb_cfo-CFO-0 2>./stdout/err_ricc-xgb_cfo-CFO-0"
sleep 10s
screen -Sdm ricc-xgb_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d riccardo  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-xgb_cfo-BS+O-0 2>./stdout/err_ricc-xgb_cfo-BS+O-0"
sleep 10s
screen -Sdm ricc-xgb_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d riccardo  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-xgb_cfo-Optu-0 2>./stdout/err_ricc-xgb_cfo-Optu-0"
sleep 10s
screen -Sdm ricc-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d riccardo  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-xgb_hpolib-Neve-0 2>./stdout/err_ricc-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm ricc-xgb_hpolib-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d riccardo  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-xgb_hpolib-Ax-0 2>./stdout/err_ricc-xgb_hpolib-Ax-0"
sleep 10s
screen -Sdm ricc-xgb_hpolib-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d riccardo  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-xgb_hpolib-Hype-0 2>./stdout/err_ricc-xgb_hpolib-Hype-0"
sleep 10s
screen -Sdm ricc-xgb_hpolib-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d riccardo  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-xgb_hpolib-CFO-0 2>./stdout/err_ricc-xgb_hpolib-CFO-0"
sleep 10s
screen -Sdm ricc-xgb_hpolib-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d riccardo  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-xgb_hpolib-BS+O-0 2>./stdout/err_ricc-xgb_hpolib-BS+O-0"
sleep 10s
screen -Sdm ricc-xgb_hpolib-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d riccardo  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-xgb_hpolib-Optu-0 2>./stdout/err_ricc-xgb_hpolib-Optu-0"
sleep 10s
screen -Sdm ricc-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d riccardo  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-xgb_blendsearch-Neve-0 2>./stdout/err_ricc-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm ricc-xgb_blendsearch-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d riccardo  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-xgb_blendsearch-Ax-0 2>./stdout/err_ricc-xgb_blendsearch-Ax-0"
sleep 10s
screen -Sdm ricc-xgb_blendsearch-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d riccardo  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-xgb_blendsearch-Hype-0 2>./stdout/err_ricc-xgb_blendsearch-Hype-0"
sleep 10s
screen -Sdm ricc-xgb_blendsearch-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d riccardo  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-xgb_blendsearch-CFO-0 2>./stdout/err_ricc-xgb_blendsearch-CFO-0"
sleep 10s
screen -Sdm ricc-xgb_blendsearch-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d riccardo  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-xgb_blendsearch-BS+O-0 2>./stdout/err_ricc-xgb_blendsearch-BS+O-0"
sleep 10s
screen -Sdm ricc-xgb_blendsearch-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d riccardo  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-xgb_blendsearch-Optu-0 2>./stdout/err_ricc-xgb_blendsearch-Optu-0"
sleep 10s
screen -Sdm higg-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d higgs  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-xgb_cfo-Neve-0 2>./stdout/err_higg-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm higg-xgb_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d higgs  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-xgb_cfo-Ax-0 2>./stdout/err_higg-xgb_cfo-Ax-0"
sleep 10s
screen -Sdm higg-xgb_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d higgs  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-xgb_cfo-Hype-0 2>./stdout/err_higg-xgb_cfo-Hype-0"
sleep 10s
screen -Sdm higg-xgb_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d higgs  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-xgb_cfo-CFO-0 2>./stdout/err_higg-xgb_cfo-CFO-0"
sleep 10s
screen -Sdm higg-xgb_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d higgs  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-xgb_cfo-BS+O-0 2>./stdout/err_higg-xgb_cfo-BS+O-0"
sleep 10s
screen -Sdm higg-xgb_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d higgs  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-xgb_cfo-Optu-0 2>./stdout/err_higg-xgb_cfo-Optu-0"
sleep 10s
sleep 15840s
screen -Sdm higg-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d higgs  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-xgb_hpolib-Neve-0 2>./stdout/err_higg-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm higg-xgb_hpolib-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d higgs  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-xgb_hpolib-Ax-0 2>./stdout/err_higg-xgb_hpolib-Ax-0"
sleep 10s
screen -Sdm higg-xgb_hpolib-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d higgs  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-xgb_hpolib-Hype-0 2>./stdout/err_higg-xgb_hpolib-Hype-0"
sleep 10s
screen -Sdm higg-xgb_hpolib-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d higgs  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-xgb_hpolib-CFO-0 2>./stdout/err_higg-xgb_hpolib-CFO-0"
sleep 10s
screen -Sdm higg-xgb_hpolib-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d higgs  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-xgb_hpolib-BS+O-0 2>./stdout/err_higg-xgb_hpolib-BS+O-0"
sleep 10s
screen -Sdm higg-xgb_hpolib-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d higgs  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-xgb_hpolib-Optu-0 2>./stdout/err_higg-xgb_hpolib-Optu-0"
sleep 10s
screen -Sdm higg-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d higgs  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-xgb_blendsearch-Neve-0 2>./stdout/err_higg-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm higg-xgb_blendsearch-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d higgs  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-xgb_blendsearch-Ax-0 2>./stdout/err_higg-xgb_blendsearch-Ax-0"
sleep 10s
screen -Sdm higg-xgb_blendsearch-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d higgs  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-xgb_blendsearch-Hype-0 2>./stdout/err_higg-xgb_blendsearch-Hype-0"
sleep 10s
screen -Sdm higg-xgb_blendsearch-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d higgs  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-xgb_blendsearch-CFO-0 2>./stdout/err_higg-xgb_blendsearch-CFO-0"
sleep 10s
screen -Sdm higg-xgb_blendsearch-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d higgs  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-xgb_blendsearch-BS+O-0 2>./stdout/err_higg-xgb_blendsearch-BS+O-0"
sleep 10s
screen -Sdm higg-xgb_blendsearch-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d higgs  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-xgb_blendsearch-Optu-0 2>./stdout/err_higg-xgb_blendsearch-Optu-0"
sleep 10s
screen -Sdm fabe-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d fabert  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-xgb_cfo-Neve-0 2>./stdout/err_fabe-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm fabe-xgb_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d fabert  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-xgb_cfo-Ax-0 2>./stdout/err_fabe-xgb_cfo-Ax-0"
sleep 10s
screen -Sdm fabe-xgb_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d fabert  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-xgb_cfo-Hype-0 2>./stdout/err_fabe-xgb_cfo-Hype-0"
sleep 10s
screen -Sdm fabe-xgb_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d fabert  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-xgb_cfo-CFO-0 2>./stdout/err_fabe-xgb_cfo-CFO-0"
sleep 10s
screen -Sdm fabe-xgb_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d fabert  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-xgb_cfo-BS+O-0 2>./stdout/err_fabe-xgb_cfo-BS+O-0"
sleep 10s
screen -Sdm fabe-xgb_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d fabert  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-xgb_cfo-Optu-0 2>./stdout/err_fabe-xgb_cfo-Optu-0"
sleep 10s
screen -Sdm fabe-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d fabert  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-xgb_hpolib-Neve-0 2>./stdout/err_fabe-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm fabe-xgb_hpolib-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d fabert  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-xgb_hpolib-Ax-0 2>./stdout/err_fabe-xgb_hpolib-Ax-0"
sleep 10s
screen -Sdm fabe-xgb_hpolib-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d fabert  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-xgb_hpolib-Hype-0 2>./stdout/err_fabe-xgb_hpolib-Hype-0"
sleep 10s
screen -Sdm fabe-xgb_hpolib-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d fabert  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-xgb_hpolib-CFO-0 2>./stdout/err_fabe-xgb_hpolib-CFO-0"
sleep 10s
screen -Sdm fabe-xgb_hpolib-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d fabert  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-xgb_hpolib-BS+O-0 2>./stdout/err_fabe-xgb_hpolib-BS+O-0"
sleep 10s
screen -Sdm fabe-xgb_hpolib-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d fabert  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-xgb_hpolib-Optu-0 2>./stdout/err_fabe-xgb_hpolib-Optu-0"
sleep 10s
screen -Sdm fabe-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d fabert  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-xgb_blendsearch-Neve-0 2>./stdout/err_fabe-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm fabe-xgb_blendsearch-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d fabert  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-xgb_blendsearch-Ax-0 2>./stdout/err_fabe-xgb_blendsearch-Ax-0"
sleep 10s
screen -Sdm fabe-xgb_blendsearch-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d fabert  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-xgb_blendsearch-Hype-0 2>./stdout/err_fabe-xgb_blendsearch-Hype-0"
sleep 10s
screen -Sdm fabe-xgb_blendsearch-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d fabert  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-xgb_blendsearch-CFO-0 2>./stdout/err_fabe-xgb_blendsearch-CFO-0"
sleep 10s
screen -Sdm fabe-xgb_blendsearch-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d fabert  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-xgb_blendsearch-BS+O-0 2>./stdout/err_fabe-xgb_blendsearch-BS+O-0"
sleep 10s
screen -Sdm fabe-xgb_blendsearch-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d fabert  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-xgb_blendsearch-Optu-0 2>./stdout/err_fabe-xgb_blendsearch-Optu-0"
sleep 10s
screen -Sdm cnae-xgb_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d cnae  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-xgb_cfo-Neve-0 2>./stdout/err_cnae-xgb_cfo-Neve-0"
sleep 10s
screen -Sdm cnae-xgb_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d cnae  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-xgb_cfo-Ax-0 2>./stdout/err_cnae-xgb_cfo-Ax-0"
sleep 10s
screen -Sdm cnae-xgb_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d cnae  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-xgb_cfo-Hype-0 2>./stdout/err_cnae-xgb_cfo-Hype-0"
sleep 10s
screen -Sdm cnae-xgb_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d cnae  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-xgb_cfo-CFO-0 2>./stdout/err_cnae-xgb_cfo-CFO-0"
sleep 10s
screen -Sdm cnae-xgb_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d cnae  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-xgb_cfo-BS+O-0 2>./stdout/err_cnae-xgb_cfo-BS+O-0"
sleep 10s
screen -Sdm cnae-xgb_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_cfo -d cnae  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-xgb_cfo-Optu-0 2>./stdout/err_cnae-xgb_cfo-Optu-0"
sleep 10s
screen -Sdm cnae-xgb_hpolib-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d cnae  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-xgb_hpolib-Neve-0 2>./stdout/err_cnae-xgb_hpolib-Neve-0"
sleep 10s
screen -Sdm cnae-xgb_hpolib-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d cnae  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-xgb_hpolib-Ax-0 2>./stdout/err_cnae-xgb_hpolib-Ax-0"
sleep 10s
screen -Sdm cnae-xgb_hpolib-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d cnae  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-xgb_hpolib-Hype-0 2>./stdout/err_cnae-xgb_hpolib-Hype-0"
sleep 10s
screen -Sdm cnae-xgb_hpolib-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d cnae  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-xgb_hpolib-CFO-0 2>./stdout/err_cnae-xgb_hpolib-CFO-0"
sleep 10s
screen -Sdm cnae-xgb_hpolib-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d cnae  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-xgb_hpolib-BS+O-0 2>./stdout/err_cnae-xgb_hpolib-BS+O-0"
sleep 10s
screen -Sdm cnae-xgb_hpolib-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_hpolib -d cnae  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-xgb_hpolib-Optu-0 2>./stdout/err_cnae-xgb_hpolib-Optu-0"
sleep 10s
screen -Sdm cnae-xgb_blendsearch-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d cnae  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-xgb_blendsearch-Neve-0 2>./stdout/err_cnae-xgb_blendsearch-Neve-0"
sleep 10s
screen -Sdm cnae-xgb_blendsearch-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d cnae  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-xgb_blendsearch-Ax-0 2>./stdout/err_cnae-xgb_blendsearch-Ax-0"
sleep 10s
screen -Sdm cnae-xgb_blendsearch-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d cnae  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-xgb_blendsearch-Hype-0 2>./stdout/err_cnae-xgb_blendsearch-Hype-0"
sleep 10s
screen -Sdm cnae-xgb_blendsearch-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d cnae  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-xgb_blendsearch-CFO-0 2>./stdout/err_cnae-xgb_blendsearch-CFO-0"
sleep 10s
screen -Sdm cnae-xgb_blendsearch-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d cnae  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-xgb_blendsearch-BS+O-0 2>./stdout/err_cnae-xgb_blendsearch-BS+O-0"
sleep 10s
screen -Sdm cnae-xgb_blendsearch-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l xgb_blendsearch -d cnae  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-xgb_blendsearch-Optu-0 2>./stdout/err_cnae-xgb_blendsearch-Optu-0"
sleep 10s