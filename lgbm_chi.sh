
screen -Sdm bloo-lgbm_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo -d blood  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-lgbm_cfo-Neve-0 2>./stdout/err_bloo-lgbm_cfo-Neve-0"
sleep 10s
screen -Sdm bloo-lgbm_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo -d blood  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-lgbm_cfo-Ax-0 2>./stdout/err_bloo-lgbm_cfo-Ax-0"
sleep 10s
screen -Sdm bloo-lgbm_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo -d blood  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-lgbm_cfo-Hype-0 2>./stdout/err_bloo-lgbm_cfo-Hype-0"
sleep 10s
screen -Sdm bloo-lgbm_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo -d blood  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-lgbm_cfo-CFO-0 2>./stdout/err_bloo-lgbm_cfo-CFO-0"
sleep 10s
screen -Sdm bloo-lgbm_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo -d blood  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-lgbm_cfo-BS+O-0 2>./stdout/err_bloo-lgbm_cfo-BS+O-0"
sleep 10s
screen -Sdm bloo-lgbm_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo -d blood  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-lgbm_cfo-Optu-0 2>./stdout/err_bloo-lgbm_cfo-Optu-0"
sleep 10s
screen -Sdm bloo-lgbm_cfo_large-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo_large -d blood  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-lgbm_cfo_large-Neve-0 2>./stdout/err_bloo-lgbm_cfo_large-Neve-0"
sleep 10s
screen -Sdm bloo-lgbm_cfo_large-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo_large -d blood  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-lgbm_cfo_large-Ax-0 2>./stdout/err_bloo-lgbm_cfo_large-Ax-0"
sleep 10s
screen -Sdm bloo-lgbm_cfo_large-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo_large -d blood  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-lgbm_cfo_large-Hype-0 2>./stdout/err_bloo-lgbm_cfo_large-Hype-0"
sleep 10s
screen -Sdm bloo-lgbm_cfo_large-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo_large -d blood  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-lgbm_cfo_large-CFO-0 2>./stdout/err_bloo-lgbm_cfo_large-CFO-0"
sleep 10s
screen -Sdm bloo-lgbm_cfo_large-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo_large -d blood  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-lgbm_cfo_large-BS+O-0 2>./stdout/err_bloo-lgbm_cfo_large-BS+O-0"
sleep 10s
screen -Sdm bloo-lgbm_cfo_large-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo_large -d blood  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_bloo-lgbm_cfo_large-Optu-0 2>./stdout/err_bloo-lgbm_cfo_large-Optu-0"
sleep 10s
screen -Sdm Aust-lgbm_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo -d Australian  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-lgbm_cfo-Neve-0 2>./stdout/err_Aust-lgbm_cfo-Neve-0"
sleep 10s
screen -Sdm Aust-lgbm_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo -d Australian  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-lgbm_cfo-Ax-0 2>./stdout/err_Aust-lgbm_cfo-Ax-0"
sleep 10s
screen -Sdm Aust-lgbm_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo -d Australian  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-lgbm_cfo-Hype-0 2>./stdout/err_Aust-lgbm_cfo-Hype-0"
sleep 10s
screen -Sdm Aust-lgbm_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo -d Australian  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-lgbm_cfo-CFO-0 2>./stdout/err_Aust-lgbm_cfo-CFO-0"
sleep 10s
screen -Sdm Aust-lgbm_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo -d Australian  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-lgbm_cfo-BS+O-0 2>./stdout/err_Aust-lgbm_cfo-BS+O-0"
sleep 10s
screen -Sdm Aust-lgbm_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo -d Australian  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-lgbm_cfo-Optu-0 2>./stdout/err_Aust-lgbm_cfo-Optu-0"
sleep 10s
screen -Sdm Aust-lgbm_cfo_large-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo_large -d Australian  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-lgbm_cfo_large-Neve-0 2>./stdout/err_Aust-lgbm_cfo_large-Neve-0"
sleep 10s
screen -Sdm Aust-lgbm_cfo_large-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo_large -d Australian  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-lgbm_cfo_large-Ax-0 2>./stdout/err_Aust-lgbm_cfo_large-Ax-0"
sleep 10s
screen -Sdm Aust-lgbm_cfo_large-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo_large -d Australian  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-lgbm_cfo_large-Hype-0 2>./stdout/err_Aust-lgbm_cfo_large-Hype-0"
sleep 10s
screen -Sdm Aust-lgbm_cfo_large-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo_large -d Australian  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-lgbm_cfo_large-CFO-0 2>./stdout/err_Aust-lgbm_cfo_large-CFO-0"
sleep 10s
screen -Sdm Aust-lgbm_cfo_large-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo_large -d Australian  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-lgbm_cfo_large-BS+O-0 2>./stdout/err_Aust-lgbm_cfo_large-BS+O-0"
sleep 10s
screen -Sdm Aust-lgbm_cfo_large-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo_large -d Australian  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Aust-lgbm_cfo_large-Optu-0 2>./stdout/err_Aust-lgbm_cfo_large-Optu-0"
sleep 10s
screen -Sdm kr-lgbm_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo -d kr  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-lgbm_cfo-Neve-0 2>./stdout/err_kr-lgbm_cfo-Neve-0"
sleep 10s
screen -Sdm kr-lgbm_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo -d kr  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-lgbm_cfo-Ax-0 2>./stdout/err_kr-lgbm_cfo-Ax-0"
sleep 10s
screen -Sdm kr-lgbm_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo -d kr  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-lgbm_cfo-Hype-0 2>./stdout/err_kr-lgbm_cfo-Hype-0"
sleep 10s
screen -Sdm kr-lgbm_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo -d kr  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-lgbm_cfo-CFO-0 2>./stdout/err_kr-lgbm_cfo-CFO-0"
sleep 10s
screen -Sdm kr-lgbm_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo -d kr  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-lgbm_cfo-BS+O-0 2>./stdout/err_kr-lgbm_cfo-BS+O-0"
sleep 10s
screen -Sdm kr-lgbm_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo -d kr  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-lgbm_cfo-Optu-0 2>./stdout/err_kr-lgbm_cfo-Optu-0"
sleep 10s
screen -Sdm kr-lgbm_cfo_large-Neve-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo_large -d kr  -m Nevergrad -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-lgbm_cfo_large-Neve-0 2>./stdout/err_kr-lgbm_cfo_large-Neve-0"
sleep 10s
screen -Sdm kr-lgbm_cfo_large-Ax-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo_large -d kr  -m Ax -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-lgbm_cfo_large-Ax-0 2>./stdout/err_kr-lgbm_cfo_large-Ax-0"
sleep 10s
screen -Sdm kr-lgbm_cfo_large-Hype-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo_large -d kr  -m HyperOpt -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-lgbm_cfo_large-Hype-0 2>./stdout/err_kr-lgbm_cfo_large-Hype-0"
sleep 10s
screen -Sdm kr-lgbm_cfo_large-CFO-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo_large -d kr  -m CFO -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-lgbm_cfo_large-CFO-0 2>./stdout/err_kr-lgbm_cfo_large-CFO-0"
sleep 10s
screen -Sdm kr-lgbm_cfo_large-BS+O-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo_large -d kr  -m BlendSearch+Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-lgbm_cfo_large-BS+O-0 2>./stdout/err_kr-lgbm_cfo_large-BS+O-0"
sleep 10s
screen -Sdm kr-lgbm_cfo_large-Optu-0 bash -c "python test/test_automl_exp.py  -t 3600.0 -l lgbm_cfo_large -d kr  -m Optuna -t 3600.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kr-lgbm_cfo_large-Optu-0 2>./stdout/err_kr-lgbm_cfo_large-Optu-0"
sleep 10s
sleep 4000s

screen -Sdm Airl-lgbm_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d Airlines  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-lgbm_cfo-Neve-0 2>./stdout/err_Airl-lgbm_cfo-Neve-0"
sleep 10s
screen -Sdm Airl-lgbm_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d Airlines  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-lgbm_cfo-Ax-0 2>./stdout/err_Airl-lgbm_cfo-Ax-0"
sleep 10s
screen -Sdm Airl-lgbm_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d Airlines  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-lgbm_cfo-Hype-0 2>./stdout/err_Airl-lgbm_cfo-Hype-0"
sleep 10s
screen -Sdm Airl-lgbm_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d Airlines  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-lgbm_cfo-CFO-0 2>./stdout/err_Airl-lgbm_cfo-CFO-0"
sleep 10s
screen -Sdm Airl-lgbm_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d Airlines  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-lgbm_cfo-BS+O-0 2>./stdout/err_Airl-lgbm_cfo-BS+O-0"
sleep 10s
screen -Sdm Airl-lgbm_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d Airlines  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-lgbm_cfo-Optu-0 2>./stdout/err_Airl-lgbm_cfo-Optu-0"
sleep 10s
screen -Sdm Airl-lgbm_cfo_large-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d Airlines  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-lgbm_cfo_large-Neve-0 2>./stdout/err_Airl-lgbm_cfo_large-Neve-0"
sleep 10s
screen -Sdm Airl-lgbm_cfo_large-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d Airlines  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-lgbm_cfo_large-Ax-0 2>./stdout/err_Airl-lgbm_cfo_large-Ax-0"
sleep 10s
screen -Sdm Airl-lgbm_cfo_large-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d Airlines  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-lgbm_cfo_large-Hype-0 2>./stdout/err_Airl-lgbm_cfo_large-Hype-0"
sleep 10s
screen -Sdm Airl-lgbm_cfo_large-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d Airlines  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-lgbm_cfo_large-CFO-0 2>./stdout/err_Airl-lgbm_cfo_large-CFO-0"
sleep 10s
screen -Sdm Airl-lgbm_cfo_large-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d Airlines  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-lgbm_cfo_large-BS+O-0 2>./stdout/err_Airl-lgbm_cfo_large-BS+O-0"
sleep 10s
screen -Sdm Airl-lgbm_cfo_large-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d Airlines  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Airl-lgbm_cfo_large-Optu-0 2>./stdout/err_Airl-lgbm_cfo_large-Optu-0"
sleep 10s
screen -Sdm chri-lgbm_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d christine  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-lgbm_cfo-Neve-0 2>./stdout/err_chri-lgbm_cfo-Neve-0"
sleep 10s
screen -Sdm chri-lgbm_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d christine  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-lgbm_cfo-Ax-0 2>./stdout/err_chri-lgbm_cfo-Ax-0"
sleep 10s
screen -Sdm chri-lgbm_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d christine  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-lgbm_cfo-Hype-0 2>./stdout/err_chri-lgbm_cfo-Hype-0"
sleep 10s
screen -Sdm chri-lgbm_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d christine  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-lgbm_cfo-CFO-0 2>./stdout/err_chri-lgbm_cfo-CFO-0"
sleep 10s
screen -Sdm chri-lgbm_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d christine  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-lgbm_cfo-BS+O-0 2>./stdout/err_chri-lgbm_cfo-BS+O-0"
sleep 10s
screen -Sdm chri-lgbm_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d christine  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-lgbm_cfo-Optu-0 2>./stdout/err_chri-lgbm_cfo-Optu-0"
sleep 10s
screen -Sdm chri-lgbm_cfo_large-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d christine  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-lgbm_cfo_large-Neve-0 2>./stdout/err_chri-lgbm_cfo_large-Neve-0"
sleep 10s
screen -Sdm chri-lgbm_cfo_large-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d christine  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-lgbm_cfo_large-Ax-0 2>./stdout/err_chri-lgbm_cfo_large-Ax-0"
sleep 10s
screen -Sdm chri-lgbm_cfo_large-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d christine  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-lgbm_cfo_large-Hype-0 2>./stdout/err_chri-lgbm_cfo_large-Hype-0"
sleep 10s
screen -Sdm chri-lgbm_cfo_large-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d christine  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-lgbm_cfo_large-CFO-0 2>./stdout/err_chri-lgbm_cfo_large-CFO-0"
sleep 10s
screen -Sdm chri-lgbm_cfo_large-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d christine  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-lgbm_cfo_large-BS+O-0 2>./stdout/err_chri-lgbm_cfo_large-BS+O-0"
sleep 10s
screen -Sdm chri-lgbm_cfo_large-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d christine  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_chri-lgbm_cfo_large-Optu-0 2>./stdout/err_chri-lgbm_cfo_large-Optu-0"
sleep 10s
screen -Sdm shut-lgbm_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d shuttle  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-lgbm_cfo-Neve-0 2>./stdout/err_shut-lgbm_cfo-Neve-0"
sleep 10s
screen -Sdm shut-lgbm_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d shuttle  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-lgbm_cfo-Ax-0 2>./stdout/err_shut-lgbm_cfo-Ax-0"
sleep 10s
screen -Sdm shut-lgbm_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d shuttle  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-lgbm_cfo-Hype-0 2>./stdout/err_shut-lgbm_cfo-Hype-0"
sleep 10s
screen -Sdm shut-lgbm_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d shuttle  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-lgbm_cfo-CFO-0 2>./stdout/err_shut-lgbm_cfo-CFO-0"
sleep 10s
screen -Sdm shut-lgbm_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d shuttle  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-lgbm_cfo-BS+O-0 2>./stdout/err_shut-lgbm_cfo-BS+O-0"
sleep 10s
screen -Sdm shut-lgbm_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d shuttle  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-lgbm_cfo-Optu-0 2>./stdout/err_shut-lgbm_cfo-Optu-0"
sleep 10s
screen -Sdm shut-lgbm_cfo_large-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d shuttle  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-lgbm_cfo_large-Neve-0 2>./stdout/err_shut-lgbm_cfo_large-Neve-0"
sleep 10s
screen -Sdm shut-lgbm_cfo_large-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d shuttle  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-lgbm_cfo_large-Ax-0 2>./stdout/err_shut-lgbm_cfo_large-Ax-0"
sleep 10s
screen -Sdm shut-lgbm_cfo_large-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d shuttle  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-lgbm_cfo_large-Hype-0 2>./stdout/err_shut-lgbm_cfo_large-Hype-0"
sleep 10s
screen -Sdm shut-lgbm_cfo_large-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d shuttle  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-lgbm_cfo_large-CFO-0 2>./stdout/err_shut-lgbm_cfo_large-CFO-0"
sleep 10s
screen -Sdm shut-lgbm_cfo_large-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d shuttle  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-lgbm_cfo_large-BS+O-0 2>./stdout/err_shut-lgbm_cfo_large-BS+O-0"
sleep 10s
screen -Sdm shut-lgbm_cfo_large-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d shuttle  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_shut-lgbm_cfo_large-Optu-0 2>./stdout/err_shut-lgbm_cfo_large-Optu-0"
sleep 10s
screen -Sdm conn-lgbm_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d connect  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-lgbm_cfo-Neve-0 2>./stdout/err_conn-lgbm_cfo-Neve-0"
sleep 10s
screen -Sdm conn-lgbm_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d connect  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-lgbm_cfo-Ax-0 2>./stdout/err_conn-lgbm_cfo-Ax-0"
sleep 10s
screen -Sdm conn-lgbm_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d connect  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-lgbm_cfo-Hype-0 2>./stdout/err_conn-lgbm_cfo-Hype-0"
sleep 10s
screen -Sdm conn-lgbm_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d connect  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-lgbm_cfo-CFO-0 2>./stdout/err_conn-lgbm_cfo-CFO-0"
sleep 10s
screen -Sdm conn-lgbm_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d connect  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-lgbm_cfo-BS+O-0 2>./stdout/err_conn-lgbm_cfo-BS+O-0"
sleep 10s
screen -Sdm conn-lgbm_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d connect  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-lgbm_cfo-Optu-0 2>./stdout/err_conn-lgbm_cfo-Optu-0"
sleep 10s
screen -Sdm conn-lgbm_cfo_large-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d connect  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-lgbm_cfo_large-Neve-0 2>./stdout/err_conn-lgbm_cfo_large-Neve-0"
sleep 10s
screen -Sdm conn-lgbm_cfo_large-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d connect  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-lgbm_cfo_large-Ax-0 2>./stdout/err_conn-lgbm_cfo_large-Ax-0"
sleep 10s
screen -Sdm conn-lgbm_cfo_large-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d connect  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-lgbm_cfo_large-Hype-0 2>./stdout/err_conn-lgbm_cfo_large-Hype-0"
sleep 10s
screen -Sdm conn-lgbm_cfo_large-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d connect  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-lgbm_cfo_large-CFO-0 2>./stdout/err_conn-lgbm_cfo_large-CFO-0"
sleep 10s
screen -Sdm conn-lgbm_cfo_large-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d connect  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-lgbm_cfo_large-BS+O-0 2>./stdout/err_conn-lgbm_cfo_large-BS+O-0"
sleep 10s
screen -Sdm conn-lgbm_cfo_large-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d connect  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_conn-lgbm_cfo_large-Optu-0 2>./stdout/err_conn-lgbm_cfo_large-Optu-0"
sleep 10s
sleep 15840s
screen -Sdm sylv-lgbm_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d sylvine  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-lgbm_cfo-Neve-0 2>./stdout/err_sylv-lgbm_cfo-Neve-0"
sleep 10s
screen -Sdm sylv-lgbm_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d sylvine  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-lgbm_cfo-Ax-0 2>./stdout/err_sylv-lgbm_cfo-Ax-0"
sleep 10s
screen -Sdm sylv-lgbm_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d sylvine  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-lgbm_cfo-Hype-0 2>./stdout/err_sylv-lgbm_cfo-Hype-0"
sleep 10s
screen -Sdm sylv-lgbm_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d sylvine  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-lgbm_cfo-CFO-0 2>./stdout/err_sylv-lgbm_cfo-CFO-0"
sleep 10s
screen -Sdm sylv-lgbm_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d sylvine  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-lgbm_cfo-BS+O-0 2>./stdout/err_sylv-lgbm_cfo-BS+O-0"
sleep 10s
screen -Sdm sylv-lgbm_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d sylvine  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-lgbm_cfo-Optu-0 2>./stdout/err_sylv-lgbm_cfo-Optu-0"
sleep 10s
screen -Sdm sylv-lgbm_cfo_large-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d sylvine  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-lgbm_cfo_large-Neve-0 2>./stdout/err_sylv-lgbm_cfo_large-Neve-0"
sleep 10s
screen -Sdm sylv-lgbm_cfo_large-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d sylvine  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-lgbm_cfo_large-Ax-0 2>./stdout/err_sylv-lgbm_cfo_large-Ax-0"
sleep 10s
screen -Sdm sylv-lgbm_cfo_large-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d sylvine  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-lgbm_cfo_large-Hype-0 2>./stdout/err_sylv-lgbm_cfo_large-Hype-0"
sleep 10s
screen -Sdm sylv-lgbm_cfo_large-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d sylvine  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-lgbm_cfo_large-CFO-0 2>./stdout/err_sylv-lgbm_cfo_large-CFO-0"
sleep 10s
screen -Sdm sylv-lgbm_cfo_large-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d sylvine  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-lgbm_cfo_large-BS+O-0 2>./stdout/err_sylv-lgbm_cfo_large-BS+O-0"
sleep 10s
screen -Sdm sylv-lgbm_cfo_large-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d sylvine  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_sylv-lgbm_cfo_large-Optu-0 2>./stdout/err_sylv-lgbm_cfo_large-Optu-0"
sleep 10s
screen -Sdm guil-lgbm_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d guillermo  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-lgbm_cfo-Neve-0 2>./stdout/err_guil-lgbm_cfo-Neve-0"
sleep 10s
screen -Sdm guil-lgbm_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d guillermo  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-lgbm_cfo-Ax-0 2>./stdout/err_guil-lgbm_cfo-Ax-0"
sleep 10s
screen -Sdm guil-lgbm_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d guillermo  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-lgbm_cfo-Hype-0 2>./stdout/err_guil-lgbm_cfo-Hype-0"
sleep 10s
screen -Sdm guil-lgbm_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d guillermo  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-lgbm_cfo-CFO-0 2>./stdout/err_guil-lgbm_cfo-CFO-0"
sleep 10s
screen -Sdm guil-lgbm_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d guillermo  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-lgbm_cfo-BS+O-0 2>./stdout/err_guil-lgbm_cfo-BS+O-0"
sleep 10s
screen -Sdm guil-lgbm_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d guillermo  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-lgbm_cfo-Optu-0 2>./stdout/err_guil-lgbm_cfo-Optu-0"
sleep 10s
screen -Sdm guil-lgbm_cfo_large-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d guillermo  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-lgbm_cfo_large-Neve-0 2>./stdout/err_guil-lgbm_cfo_large-Neve-0"
sleep 10s
screen -Sdm guil-lgbm_cfo_large-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d guillermo  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-lgbm_cfo_large-Ax-0 2>./stdout/err_guil-lgbm_cfo_large-Ax-0"
sleep 10s
screen -Sdm guil-lgbm_cfo_large-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d guillermo  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-lgbm_cfo_large-Hype-0 2>./stdout/err_guil-lgbm_cfo_large-Hype-0"
sleep 10s
screen -Sdm guil-lgbm_cfo_large-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d guillermo  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-lgbm_cfo_large-CFO-0 2>./stdout/err_guil-lgbm_cfo_large-CFO-0"
sleep 10s
screen -Sdm guil-lgbm_cfo_large-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d guillermo  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-lgbm_cfo_large-BS+O-0 2>./stdout/err_guil-lgbm_cfo_large-BS+O-0"
sleep 10s
screen -Sdm guil-lgbm_cfo_large-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d guillermo  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_guil-lgbm_cfo_large-Optu-0 2>./stdout/err_guil-lgbm_cfo_large-Optu-0"
sleep 10s
screen -Sdm volk-lgbm_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d volkert  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-lgbm_cfo-Neve-0 2>./stdout/err_volk-lgbm_cfo-Neve-0"
sleep 10s
screen -Sdm volk-lgbm_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d volkert  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-lgbm_cfo-Ax-0 2>./stdout/err_volk-lgbm_cfo-Ax-0"
sleep 10s
screen -Sdm volk-lgbm_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d volkert  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-lgbm_cfo-Hype-0 2>./stdout/err_volk-lgbm_cfo-Hype-0"
sleep 10s
screen -Sdm volk-lgbm_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d volkert  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-lgbm_cfo-CFO-0 2>./stdout/err_volk-lgbm_cfo-CFO-0"
sleep 10s
screen -Sdm volk-lgbm_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d volkert  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-lgbm_cfo-BS+O-0 2>./stdout/err_volk-lgbm_cfo-BS+O-0"
sleep 10s
screen -Sdm volk-lgbm_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d volkert  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-lgbm_cfo-Optu-0 2>./stdout/err_volk-lgbm_cfo-Optu-0"
sleep 10s
screen -Sdm volk-lgbm_cfo_large-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d volkert  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-lgbm_cfo_large-Neve-0 2>./stdout/err_volk-lgbm_cfo_large-Neve-0"
sleep 10s
screen -Sdm volk-lgbm_cfo_large-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d volkert  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-lgbm_cfo_large-Ax-0 2>./stdout/err_volk-lgbm_cfo_large-Ax-0"
sleep 10s
screen -Sdm volk-lgbm_cfo_large-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d volkert  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-lgbm_cfo_large-Hype-0 2>./stdout/err_volk-lgbm_cfo_large-Hype-0"
sleep 10s
screen -Sdm volk-lgbm_cfo_large-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d volkert  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-lgbm_cfo_large-CFO-0 2>./stdout/err_volk-lgbm_cfo_large-CFO-0"
sleep 10s
screen -Sdm volk-lgbm_cfo_large-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d volkert  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-lgbm_cfo_large-BS+O-0 2>./stdout/err_volk-lgbm_cfo_large-BS+O-0"
sleep 10s
screen -Sdm volk-lgbm_cfo_large-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d volkert  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_volk-lgbm_cfo_large-Optu-0 2>./stdout/err_volk-lgbm_cfo_large-Optu-0"
sleep 10s
screen -Sdm Mini-lgbm_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d MiniBooNE  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-lgbm_cfo-Neve-0 2>./stdout/err_Mini-lgbm_cfo-Neve-0"
sleep 10s
screen -Sdm Mini-lgbm_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d MiniBooNE  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-lgbm_cfo-Ax-0 2>./stdout/err_Mini-lgbm_cfo-Ax-0"
sleep 10s
screen -Sdm Mini-lgbm_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d MiniBooNE  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-lgbm_cfo-Hype-0 2>./stdout/err_Mini-lgbm_cfo-Hype-0"
sleep 10s
screen -Sdm Mini-lgbm_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d MiniBooNE  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-lgbm_cfo-CFO-0 2>./stdout/err_Mini-lgbm_cfo-CFO-0"
sleep 10s
screen -Sdm Mini-lgbm_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d MiniBooNE  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-lgbm_cfo-BS+O-0 2>./stdout/err_Mini-lgbm_cfo-BS+O-0"
sleep 10s
screen -Sdm Mini-lgbm_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d MiniBooNE  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-lgbm_cfo-Optu-0 2>./stdout/err_Mini-lgbm_cfo-Optu-0"
sleep 10s
screen -Sdm Mini-lgbm_cfo_large-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d MiniBooNE  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-lgbm_cfo_large-Neve-0 2>./stdout/err_Mini-lgbm_cfo_large-Neve-0"
sleep 10s
screen -Sdm Mini-lgbm_cfo_large-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d MiniBooNE  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-lgbm_cfo_large-Ax-0 2>./stdout/err_Mini-lgbm_cfo_large-Ax-0"
sleep 10s
screen -Sdm Mini-lgbm_cfo_large-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d MiniBooNE  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-lgbm_cfo_large-Hype-0 2>./stdout/err_Mini-lgbm_cfo_large-Hype-0"
sleep 10s
screen -Sdm Mini-lgbm_cfo_large-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d MiniBooNE  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-lgbm_cfo_large-CFO-0 2>./stdout/err_Mini-lgbm_cfo_large-CFO-0"
sleep 10s
screen -Sdm Mini-lgbm_cfo_large-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d MiniBooNE  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-lgbm_cfo_large-BS+O-0 2>./stdout/err_Mini-lgbm_cfo_large-BS+O-0"
sleep 10s
screen -Sdm Mini-lgbm_cfo_large-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d MiniBooNE  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Mini-lgbm_cfo_large-Optu-0 2>./stdout/err_Mini-lgbm_cfo_large-Optu-0"
sleep 10s
sleep 15840s
screen -Sdm Jann-lgbm_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d Jannis  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-lgbm_cfo-Neve-0 2>./stdout/err_Jann-lgbm_cfo-Neve-0"
sleep 10s
screen -Sdm Jann-lgbm_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d Jannis  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-lgbm_cfo-Ax-0 2>./stdout/err_Jann-lgbm_cfo-Ax-0"
sleep 10s
screen -Sdm Jann-lgbm_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d Jannis  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-lgbm_cfo-Hype-0 2>./stdout/err_Jann-lgbm_cfo-Hype-0"
sleep 10s
screen -Sdm Jann-lgbm_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d Jannis  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-lgbm_cfo-CFO-0 2>./stdout/err_Jann-lgbm_cfo-CFO-0"
sleep 10s
screen -Sdm Jann-lgbm_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d Jannis  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-lgbm_cfo-BS+O-0 2>./stdout/err_Jann-lgbm_cfo-BS+O-0"
sleep 10s
screen -Sdm Jann-lgbm_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d Jannis  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-lgbm_cfo-Optu-0 2>./stdout/err_Jann-lgbm_cfo-Optu-0"
sleep 10s
screen -Sdm Jann-lgbm_cfo_large-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d Jannis  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-lgbm_cfo_large-Neve-0 2>./stdout/err_Jann-lgbm_cfo_large-Neve-0"
sleep 10s
screen -Sdm Jann-lgbm_cfo_large-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d Jannis  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-lgbm_cfo_large-Ax-0 2>./stdout/err_Jann-lgbm_cfo_large-Ax-0"
sleep 10s
screen -Sdm Jann-lgbm_cfo_large-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d Jannis  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-lgbm_cfo_large-Hype-0 2>./stdout/err_Jann-lgbm_cfo_large-Hype-0"
sleep 10s
screen -Sdm Jann-lgbm_cfo_large-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d Jannis  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-lgbm_cfo_large-CFO-0 2>./stdout/err_Jann-lgbm_cfo_large-CFO-0"
sleep 10s
screen -Sdm Jann-lgbm_cfo_large-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d Jannis  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-lgbm_cfo_large-BS+O-0 2>./stdout/err_Jann-lgbm_cfo_large-BS+O-0"
sleep 10s
screen -Sdm Jann-lgbm_cfo_large-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d Jannis  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_Jann-lgbm_cfo_large-Optu-0 2>./stdout/err_Jann-lgbm_cfo_large-Optu-0"
sleep 10s
screen -Sdm mfea-lgbm_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d mfeat  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-lgbm_cfo-Neve-0 2>./stdout/err_mfea-lgbm_cfo-Neve-0"
sleep 10s
screen -Sdm mfea-lgbm_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d mfeat  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-lgbm_cfo-Ax-0 2>./stdout/err_mfea-lgbm_cfo-Ax-0"
sleep 10s
screen -Sdm mfea-lgbm_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d mfeat  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-lgbm_cfo-Hype-0 2>./stdout/err_mfea-lgbm_cfo-Hype-0"
sleep 10s
screen -Sdm mfea-lgbm_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d mfeat  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-lgbm_cfo-CFO-0 2>./stdout/err_mfea-lgbm_cfo-CFO-0"
sleep 10s
screen -Sdm mfea-lgbm_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d mfeat  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-lgbm_cfo-BS+O-0 2>./stdout/err_mfea-lgbm_cfo-BS+O-0"
sleep 10s
screen -Sdm mfea-lgbm_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d mfeat  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-lgbm_cfo-Optu-0 2>./stdout/err_mfea-lgbm_cfo-Optu-0"
sleep 10s
screen -Sdm mfea-lgbm_cfo_large-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d mfeat  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-lgbm_cfo_large-Neve-0 2>./stdout/err_mfea-lgbm_cfo_large-Neve-0"
sleep 10s
screen -Sdm mfea-lgbm_cfo_large-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d mfeat  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-lgbm_cfo_large-Ax-0 2>./stdout/err_mfea-lgbm_cfo_large-Ax-0"
sleep 10s
screen -Sdm mfea-lgbm_cfo_large-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d mfeat  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-lgbm_cfo_large-Hype-0 2>./stdout/err_mfea-lgbm_cfo_large-Hype-0"
sleep 10s
screen -Sdm mfea-lgbm_cfo_large-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d mfeat  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-lgbm_cfo_large-CFO-0 2>./stdout/err_mfea-lgbm_cfo_large-CFO-0"
sleep 10s
screen -Sdm mfea-lgbm_cfo_large-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d mfeat  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-lgbm_cfo_large-BS+O-0 2>./stdout/err_mfea-lgbm_cfo_large-BS+O-0"
sleep 10s
screen -Sdm mfea-lgbm_cfo_large-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d mfeat  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_mfea-lgbm_cfo_large-Optu-0 2>./stdout/err_mfea-lgbm_cfo_large-Optu-0"
sleep 10s
screen -Sdm jung-lgbm_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d jungle  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-lgbm_cfo-Neve-0 2>./stdout/err_jung-lgbm_cfo-Neve-0"
sleep 10s
screen -Sdm jung-lgbm_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d jungle  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-lgbm_cfo-Ax-0 2>./stdout/err_jung-lgbm_cfo-Ax-0"
sleep 10s
screen -Sdm jung-lgbm_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d jungle  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-lgbm_cfo-Hype-0 2>./stdout/err_jung-lgbm_cfo-Hype-0"
sleep 10s
screen -Sdm jung-lgbm_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d jungle  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-lgbm_cfo-CFO-0 2>./stdout/err_jung-lgbm_cfo-CFO-0"
sleep 10s
screen -Sdm jung-lgbm_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d jungle  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-lgbm_cfo-BS+O-0 2>./stdout/err_jung-lgbm_cfo-BS+O-0"
sleep 10s
screen -Sdm jung-lgbm_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d jungle  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-lgbm_cfo-Optu-0 2>./stdout/err_jung-lgbm_cfo-Optu-0"
sleep 10s
screen -Sdm jung-lgbm_cfo_large-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d jungle  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-lgbm_cfo_large-Neve-0 2>./stdout/err_jung-lgbm_cfo_large-Neve-0"
sleep 10s
screen -Sdm jung-lgbm_cfo_large-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d jungle  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-lgbm_cfo_large-Ax-0 2>./stdout/err_jung-lgbm_cfo_large-Ax-0"
sleep 10s
screen -Sdm jung-lgbm_cfo_large-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d jungle  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-lgbm_cfo_large-Hype-0 2>./stdout/err_jung-lgbm_cfo_large-Hype-0"
sleep 10s
screen -Sdm jung-lgbm_cfo_large-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d jungle  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-lgbm_cfo_large-CFO-0 2>./stdout/err_jung-lgbm_cfo_large-CFO-0"
sleep 10s
screen -Sdm jung-lgbm_cfo_large-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d jungle  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-lgbm_cfo_large-BS+O-0 2>./stdout/err_jung-lgbm_cfo_large-BS+O-0"
sleep 10s
screen -Sdm jung-lgbm_cfo_large-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d jungle  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jung-lgbm_cfo_large-Optu-0 2>./stdout/err_jung-lgbm_cfo_large-Optu-0"
sleep 10s
screen -Sdm jasm-lgbm_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d jasmine  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-lgbm_cfo-Neve-0 2>./stdout/err_jasm-lgbm_cfo-Neve-0"
sleep 10s
screen -Sdm jasm-lgbm_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d jasmine  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-lgbm_cfo-Ax-0 2>./stdout/err_jasm-lgbm_cfo-Ax-0"
sleep 10s
screen -Sdm jasm-lgbm_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d jasmine  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-lgbm_cfo-Hype-0 2>./stdout/err_jasm-lgbm_cfo-Hype-0"
sleep 10s
screen -Sdm jasm-lgbm_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d jasmine  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-lgbm_cfo-CFO-0 2>./stdout/err_jasm-lgbm_cfo-CFO-0"
sleep 10s
screen -Sdm jasm-lgbm_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d jasmine  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-lgbm_cfo-BS+O-0 2>./stdout/err_jasm-lgbm_cfo-BS+O-0"
sleep 10s
screen -Sdm jasm-lgbm_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d jasmine  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-lgbm_cfo-Optu-0 2>./stdout/err_jasm-lgbm_cfo-Optu-0"
sleep 10s
screen -Sdm jasm-lgbm_cfo_large-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d jasmine  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-lgbm_cfo_large-Neve-0 2>./stdout/err_jasm-lgbm_cfo_large-Neve-0"
sleep 10s
screen -Sdm jasm-lgbm_cfo_large-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d jasmine  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-lgbm_cfo_large-Ax-0 2>./stdout/err_jasm-lgbm_cfo_large-Ax-0"
sleep 10s
screen -Sdm jasm-lgbm_cfo_large-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d jasmine  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-lgbm_cfo_large-Hype-0 2>./stdout/err_jasm-lgbm_cfo_large-Hype-0"
sleep 10s
screen -Sdm jasm-lgbm_cfo_large-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d jasmine  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-lgbm_cfo_large-CFO-0 2>./stdout/err_jasm-lgbm_cfo_large-CFO-0"
sleep 10s
screen -Sdm jasm-lgbm_cfo_large-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d jasmine  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-lgbm_cfo_large-BS+O-0 2>./stdout/err_jasm-lgbm_cfo_large-BS+O-0"
sleep 10s
screen -Sdm jasm-lgbm_cfo_large-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d jasmine  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_jasm-lgbm_cfo_large-Optu-0 2>./stdout/err_jasm-lgbm_cfo_large-Optu-0"
sleep 10s
sleep 15840s
screen -Sdm ricc-lgbm_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d riccardo  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-lgbm_cfo-Neve-0 2>./stdout/err_ricc-lgbm_cfo-Neve-0"
sleep 10s
screen -Sdm ricc-lgbm_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d riccardo  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-lgbm_cfo-Ax-0 2>./stdout/err_ricc-lgbm_cfo-Ax-0"
sleep 10s
screen -Sdm ricc-lgbm_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d riccardo  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-lgbm_cfo-Hype-0 2>./stdout/err_ricc-lgbm_cfo-Hype-0"
sleep 10s
screen -Sdm ricc-lgbm_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d riccardo  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-lgbm_cfo-CFO-0 2>./stdout/err_ricc-lgbm_cfo-CFO-0"
sleep 10s
screen -Sdm ricc-lgbm_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d riccardo  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-lgbm_cfo-BS+O-0 2>./stdout/err_ricc-lgbm_cfo-BS+O-0"
sleep 10s
screen -Sdm ricc-lgbm_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d riccardo  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-lgbm_cfo-Optu-0 2>./stdout/err_ricc-lgbm_cfo-Optu-0"
sleep 10s
screen -Sdm ricc-lgbm_cfo_large-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d riccardo  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-lgbm_cfo_large-Neve-0 2>./stdout/err_ricc-lgbm_cfo_large-Neve-0"
sleep 10s
screen -Sdm ricc-lgbm_cfo_large-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d riccardo  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-lgbm_cfo_large-Ax-0 2>./stdout/err_ricc-lgbm_cfo_large-Ax-0"
sleep 10s
screen -Sdm ricc-lgbm_cfo_large-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d riccardo  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-lgbm_cfo_large-Hype-0 2>./stdout/err_ricc-lgbm_cfo_large-Hype-0"
sleep 10s
screen -Sdm ricc-lgbm_cfo_large-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d riccardo  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-lgbm_cfo_large-CFO-0 2>./stdout/err_ricc-lgbm_cfo_large-CFO-0"
sleep 10s
screen -Sdm ricc-lgbm_cfo_large-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d riccardo  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-lgbm_cfo_large-BS+O-0 2>./stdout/err_ricc-lgbm_cfo_large-BS+O-0"
sleep 10s
screen -Sdm ricc-lgbm_cfo_large-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d riccardo  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_ricc-lgbm_cfo_large-Optu-0 2>./stdout/err_ricc-lgbm_cfo_large-Optu-0"
sleep 10s
screen -Sdm higg-lgbm_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d higgs  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-lgbm_cfo-Neve-0 2>./stdout/err_higg-lgbm_cfo-Neve-0"
sleep 10s
screen -Sdm higg-lgbm_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d higgs  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-lgbm_cfo-Ax-0 2>./stdout/err_higg-lgbm_cfo-Ax-0"
sleep 10s
screen -Sdm higg-lgbm_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d higgs  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-lgbm_cfo-Hype-0 2>./stdout/err_higg-lgbm_cfo-Hype-0"
sleep 10s
screen -Sdm higg-lgbm_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d higgs  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-lgbm_cfo-CFO-0 2>./stdout/err_higg-lgbm_cfo-CFO-0"
sleep 10s
screen -Sdm higg-lgbm_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d higgs  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-lgbm_cfo-BS+O-0 2>./stdout/err_higg-lgbm_cfo-BS+O-0"
sleep 10s
screen -Sdm higg-lgbm_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d higgs  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-lgbm_cfo-Optu-0 2>./stdout/err_higg-lgbm_cfo-Optu-0"
sleep 10s
screen -Sdm higg-lgbm_cfo_large-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d higgs  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-lgbm_cfo_large-Neve-0 2>./stdout/err_higg-lgbm_cfo_large-Neve-0"
sleep 10s
screen -Sdm higg-lgbm_cfo_large-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d higgs  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-lgbm_cfo_large-Ax-0 2>./stdout/err_higg-lgbm_cfo_large-Ax-0"
sleep 10s
screen -Sdm higg-lgbm_cfo_large-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d higgs  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-lgbm_cfo_large-Hype-0 2>./stdout/err_higg-lgbm_cfo_large-Hype-0"
sleep 10s
screen -Sdm higg-lgbm_cfo_large-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d higgs  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-lgbm_cfo_large-CFO-0 2>./stdout/err_higg-lgbm_cfo_large-CFO-0"
sleep 10s
screen -Sdm higg-lgbm_cfo_large-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d higgs  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-lgbm_cfo_large-BS+O-0 2>./stdout/err_higg-lgbm_cfo_large-BS+O-0"
sleep 10s
screen -Sdm higg-lgbm_cfo_large-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d higgs  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_higg-lgbm_cfo_large-Optu-0 2>./stdout/err_higg-lgbm_cfo_large-Optu-0"
sleep 10s
screen -Sdm fabe-lgbm_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d fabert  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-lgbm_cfo-Neve-0 2>./stdout/err_fabe-lgbm_cfo-Neve-0"
sleep 10s
screen -Sdm fabe-lgbm_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d fabert  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-lgbm_cfo-Ax-0 2>./stdout/err_fabe-lgbm_cfo-Ax-0"
sleep 10s
screen -Sdm fabe-lgbm_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d fabert  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-lgbm_cfo-Hype-0 2>./stdout/err_fabe-lgbm_cfo-Hype-0"
sleep 10s
screen -Sdm fabe-lgbm_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d fabert  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-lgbm_cfo-CFO-0 2>./stdout/err_fabe-lgbm_cfo-CFO-0"
sleep 10s
screen -Sdm fabe-lgbm_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d fabert  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-lgbm_cfo-BS+O-0 2>./stdout/err_fabe-lgbm_cfo-BS+O-0"
sleep 10s
screen -Sdm fabe-lgbm_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d fabert  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-lgbm_cfo-Optu-0 2>./stdout/err_fabe-lgbm_cfo-Optu-0"
sleep 10s
screen -Sdm fabe-lgbm_cfo_large-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d fabert  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-lgbm_cfo_large-Neve-0 2>./stdout/err_fabe-lgbm_cfo_large-Neve-0"
sleep 10s
screen -Sdm fabe-lgbm_cfo_large-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d fabert  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-lgbm_cfo_large-Ax-0 2>./stdout/err_fabe-lgbm_cfo_large-Ax-0"
sleep 10s
screen -Sdm fabe-lgbm_cfo_large-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d fabert  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-lgbm_cfo_large-Hype-0 2>./stdout/err_fabe-lgbm_cfo_large-Hype-0"
sleep 10s
screen -Sdm fabe-lgbm_cfo_large-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d fabert  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-lgbm_cfo_large-CFO-0 2>./stdout/err_fabe-lgbm_cfo_large-CFO-0"
sleep 10s
screen -Sdm fabe-lgbm_cfo_large-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d fabert  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-lgbm_cfo_large-BS+O-0 2>./stdout/err_fabe-lgbm_cfo_large-BS+O-0"
sleep 10s
screen -Sdm fabe-lgbm_cfo_large-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d fabert  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_fabe-lgbm_cfo_large-Optu-0 2>./stdout/err_fabe-lgbm_cfo_large-Optu-0"
sleep 10s
screen -Sdm cnae-lgbm_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d cnae  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-lgbm_cfo-Neve-0 2>./stdout/err_cnae-lgbm_cfo-Neve-0"
sleep 10s
screen -Sdm cnae-lgbm_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d cnae  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-lgbm_cfo-Ax-0 2>./stdout/err_cnae-lgbm_cfo-Ax-0"
sleep 10s
screen -Sdm cnae-lgbm_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d cnae  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-lgbm_cfo-Hype-0 2>./stdout/err_cnae-lgbm_cfo-Hype-0"
sleep 10s
screen -Sdm cnae-lgbm_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d cnae  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-lgbm_cfo-CFO-0 2>./stdout/err_cnae-lgbm_cfo-CFO-0"
sleep 10s
screen -Sdm cnae-lgbm_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d cnae  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-lgbm_cfo-BS+O-0 2>./stdout/err_cnae-lgbm_cfo-BS+O-0"
sleep 10s
screen -Sdm cnae-lgbm_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d cnae  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-lgbm_cfo-Optu-0 2>./stdout/err_cnae-lgbm_cfo-Optu-0"
sleep 10s
screen -Sdm cnae-lgbm_cfo_large-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d cnae  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-lgbm_cfo_large-Neve-0 2>./stdout/err_cnae-lgbm_cfo_large-Neve-0"
sleep 10s
screen -Sdm cnae-lgbm_cfo_large-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d cnae  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-lgbm_cfo_large-Ax-0 2>./stdout/err_cnae-lgbm_cfo_large-Ax-0"
sleep 10s
screen -Sdm cnae-lgbm_cfo_large-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d cnae  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-lgbm_cfo_large-Hype-0 2>./stdout/err_cnae-lgbm_cfo_large-Hype-0"
sleep 10s
screen -Sdm cnae-lgbm_cfo_large-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d cnae  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-lgbm_cfo_large-CFO-0 2>./stdout/err_cnae-lgbm_cfo_large-CFO-0"
sleep 10s
screen -Sdm cnae-lgbm_cfo_large-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d cnae  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-lgbm_cfo_large-BS+O-0 2>./stdout/err_cnae-lgbm_cfo_large-BS+O-0"
sleep 10s
screen -Sdm cnae-lgbm_cfo_large-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d cnae  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cnae-lgbm_cfo_large-Optu-0 2>./stdout/err_cnae-lgbm_cfo_large-Optu-0"
sleep 10s
sleep 15840s
screen -Sdm KDDC-lgbm_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d KDDCup09  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_KDDC-lgbm_cfo-Neve-0 2>./stdout/err_KDDC-lgbm_cfo-Neve-0"
sleep 10s
screen -Sdm KDDC-lgbm_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d KDDCup09  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_KDDC-lgbm_cfo-Ax-0 2>./stdout/err_KDDC-lgbm_cfo-Ax-0"
sleep 10s
screen -Sdm KDDC-lgbm_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d KDDCup09  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_KDDC-lgbm_cfo-Hype-0 2>./stdout/err_KDDC-lgbm_cfo-Hype-0"
sleep 10s
screen -Sdm KDDC-lgbm_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d KDDCup09  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_KDDC-lgbm_cfo-CFO-0 2>./stdout/err_KDDC-lgbm_cfo-CFO-0"
sleep 10s
screen -Sdm KDDC-lgbm_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d KDDCup09  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_KDDC-lgbm_cfo-BS+O-0 2>./stdout/err_KDDC-lgbm_cfo-BS+O-0"
sleep 10s
screen -Sdm KDDC-lgbm_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d KDDCup09  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_KDDC-lgbm_cfo-Optu-0 2>./stdout/err_KDDC-lgbm_cfo-Optu-0"
sleep 10s
screen -Sdm KDDC-lgbm_cfo_large-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d KDDCup09  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_KDDC-lgbm_cfo_large-Neve-0 2>./stdout/err_KDDC-lgbm_cfo_large-Neve-0"
sleep 10s
screen -Sdm KDDC-lgbm_cfo_large-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d KDDCup09  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_KDDC-lgbm_cfo_large-Ax-0 2>./stdout/err_KDDC-lgbm_cfo_large-Ax-0"
sleep 10s
screen -Sdm KDDC-lgbm_cfo_large-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d KDDCup09  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_KDDC-lgbm_cfo_large-Hype-0 2>./stdout/err_KDDC-lgbm_cfo_large-Hype-0"
sleep 10s
screen -Sdm KDDC-lgbm_cfo_large-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d KDDCup09  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_KDDC-lgbm_cfo_large-CFO-0 2>./stdout/err_KDDC-lgbm_cfo_large-CFO-0"
sleep 10s
screen -Sdm KDDC-lgbm_cfo_large-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d KDDCup09  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_KDDC-lgbm_cfo_large-BS+O-0 2>./stdout/err_KDDC-lgbm_cfo_large-BS+O-0"
sleep 10s
screen -Sdm KDDC-lgbm_cfo_large-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d KDDCup09  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_KDDC-lgbm_cfo_large-Optu-0 2>./stdout/err_KDDC-lgbm_cfo_large-Optu-0"
sleep 10s
screen -Sdm nume-lgbm_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d numerai28  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_nume-lgbm_cfo-Neve-0 2>./stdout/err_nume-lgbm_cfo-Neve-0"
sleep 10s
screen -Sdm nume-lgbm_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d numerai28  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_nume-lgbm_cfo-Ax-0 2>./stdout/err_nume-lgbm_cfo-Ax-0"
sleep 10s
screen -Sdm nume-lgbm_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d numerai28  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_nume-lgbm_cfo-Hype-0 2>./stdout/err_nume-lgbm_cfo-Hype-0"
sleep 10s
screen -Sdm nume-lgbm_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d numerai28  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_nume-lgbm_cfo-CFO-0 2>./stdout/err_nume-lgbm_cfo-CFO-0"
sleep 10s
screen -Sdm nume-lgbm_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d numerai28  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_nume-lgbm_cfo-BS+O-0 2>./stdout/err_nume-lgbm_cfo-BS+O-0"
sleep 10s
screen -Sdm nume-lgbm_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d numerai28  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_nume-lgbm_cfo-Optu-0 2>./stdout/err_nume-lgbm_cfo-Optu-0"
sleep 10s
screen -Sdm nume-lgbm_cfo_large-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d numerai28  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_nume-lgbm_cfo_large-Neve-0 2>./stdout/err_nume-lgbm_cfo_large-Neve-0"
sleep 10s
screen -Sdm nume-lgbm_cfo_large-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d numerai28  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_nume-lgbm_cfo_large-Ax-0 2>./stdout/err_nume-lgbm_cfo_large-Ax-0"
sleep 10s
screen -Sdm nume-lgbm_cfo_large-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d numerai28  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_nume-lgbm_cfo_large-Hype-0 2>./stdout/err_nume-lgbm_cfo_large-Hype-0"
sleep 10s
screen -Sdm nume-lgbm_cfo_large-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d numerai28  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_nume-lgbm_cfo_large-CFO-0 2>./stdout/err_nume-lgbm_cfo_large-CFO-0"
sleep 10s
screen -Sdm nume-lgbm_cfo_large-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d numerai28  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_nume-lgbm_cfo_large-BS+O-0 2>./stdout/err_nume-lgbm_cfo_large-BS+O-0"
sleep 10s
screen -Sdm nume-lgbm_cfo_large-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d numerai28  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_nume-lgbm_cfo_large-Optu-0 2>./stdout/err_nume-lgbm_cfo_large-Optu-0"
sleep 10s
screen -Sdm cred-lgbm_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d credit  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-lgbm_cfo-Neve-0 2>./stdout/err_cred-lgbm_cfo-Neve-0"
sleep 10s
screen -Sdm cred-lgbm_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d credit  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-lgbm_cfo-Ax-0 2>./stdout/err_cred-lgbm_cfo-Ax-0"
sleep 10s
screen -Sdm cred-lgbm_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d credit  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-lgbm_cfo-Hype-0 2>./stdout/err_cred-lgbm_cfo-Hype-0"
sleep 10s
screen -Sdm cred-lgbm_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d credit  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-lgbm_cfo-CFO-0 2>./stdout/err_cred-lgbm_cfo-CFO-0"
sleep 10s
screen -Sdm cred-lgbm_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d credit  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-lgbm_cfo-BS+O-0 2>./stdout/err_cred-lgbm_cfo-BS+O-0"
sleep 10s
screen -Sdm cred-lgbm_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d credit  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-lgbm_cfo-Optu-0 2>./stdout/err_cred-lgbm_cfo-Optu-0"
sleep 10s
screen -Sdm cred-lgbm_cfo_large-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d credit  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-lgbm_cfo_large-Neve-0 2>./stdout/err_cred-lgbm_cfo_large-Neve-0"
sleep 10s
screen -Sdm cred-lgbm_cfo_large-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d credit  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-lgbm_cfo_large-Ax-0 2>./stdout/err_cred-lgbm_cfo_large-Ax-0"
sleep 10s
screen -Sdm cred-lgbm_cfo_large-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d credit  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-lgbm_cfo_large-Hype-0 2>./stdout/err_cred-lgbm_cfo_large-Hype-0"
sleep 10s
screen -Sdm cred-lgbm_cfo_large-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d credit  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-lgbm_cfo_large-CFO-0 2>./stdout/err_cred-lgbm_cfo_large-CFO-0"
sleep 10s
screen -Sdm cred-lgbm_cfo_large-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d credit  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-lgbm_cfo_large-BS+O-0 2>./stdout/err_cred-lgbm_cfo_large-BS+O-0"
sleep 10s
screen -Sdm cred-lgbm_cfo_large-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d credit  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_cred-lgbm_cfo_large-Optu-0 2>./stdout/err_cred-lgbm_cfo_large-Optu-0"
sleep 10s
screen -Sdm car-lgbm_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d car  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-lgbm_cfo-Neve-0 2>./stdout/err_car-lgbm_cfo-Neve-0"
sleep 10s
screen -Sdm car-lgbm_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d car  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-lgbm_cfo-Ax-0 2>./stdout/err_car-lgbm_cfo-Ax-0"
sleep 10s
screen -Sdm car-lgbm_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d car  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-lgbm_cfo-Hype-0 2>./stdout/err_car-lgbm_cfo-Hype-0"
sleep 10s
screen -Sdm car-lgbm_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d car  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-lgbm_cfo-CFO-0 2>./stdout/err_car-lgbm_cfo-CFO-0"
sleep 10s
screen -Sdm car-lgbm_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d car  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-lgbm_cfo-BS+O-0 2>./stdout/err_car-lgbm_cfo-BS+O-0"
sleep 10s
screen -Sdm car-lgbm_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d car  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-lgbm_cfo-Optu-0 2>./stdout/err_car-lgbm_cfo-Optu-0"
sleep 10s
screen -Sdm car-lgbm_cfo_large-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d car  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-lgbm_cfo_large-Neve-0 2>./stdout/err_car-lgbm_cfo_large-Neve-0"
sleep 10s
screen -Sdm car-lgbm_cfo_large-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d car  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-lgbm_cfo_large-Ax-0 2>./stdout/err_car-lgbm_cfo_large-Ax-0"
sleep 10s
screen -Sdm car-lgbm_cfo_large-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d car  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-lgbm_cfo_large-Hype-0 2>./stdout/err_car-lgbm_cfo_large-Hype-0"
sleep 10s
screen -Sdm car-lgbm_cfo_large-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d car  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-lgbm_cfo_large-CFO-0 2>./stdout/err_car-lgbm_cfo_large-CFO-0"
sleep 10s
screen -Sdm car-lgbm_cfo_large-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d car  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-lgbm_cfo_large-BS+O-0 2>./stdout/err_car-lgbm_cfo_large-BS+O-0"
sleep 10s
screen -Sdm car-lgbm_cfo_large-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d car  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_car-lgbm_cfo_large-Optu-0 2>./stdout/err_car-lgbm_cfo_large-Optu-0"
sleep 10s
sleep 15840s
screen -Sdm kc1-lgbm_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d kc1  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-lgbm_cfo-Neve-0 2>./stdout/err_kc1-lgbm_cfo-Neve-0"
sleep 10s
screen -Sdm kc1-lgbm_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d kc1  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-lgbm_cfo-Ax-0 2>./stdout/err_kc1-lgbm_cfo-Ax-0"
sleep 10s
screen -Sdm kc1-lgbm_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d kc1  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-lgbm_cfo-Hype-0 2>./stdout/err_kc1-lgbm_cfo-Hype-0"
sleep 10s
screen -Sdm kc1-lgbm_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d kc1  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-lgbm_cfo-CFO-0 2>./stdout/err_kc1-lgbm_cfo-CFO-0"
sleep 10s
screen -Sdm kc1-lgbm_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d kc1  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-lgbm_cfo-BS+O-0 2>./stdout/err_kc1-lgbm_cfo-BS+O-0"
sleep 10s
screen -Sdm kc1-lgbm_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d kc1  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-lgbm_cfo-Optu-0 2>./stdout/err_kc1-lgbm_cfo-Optu-0"
sleep 10s
screen -Sdm kc1-lgbm_cfo_large-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d kc1  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-lgbm_cfo_large-Neve-0 2>./stdout/err_kc1-lgbm_cfo_large-Neve-0"
sleep 10s
screen -Sdm kc1-lgbm_cfo_large-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d kc1  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-lgbm_cfo_large-Ax-0 2>./stdout/err_kc1-lgbm_cfo_large-Ax-0"
sleep 10s
screen -Sdm kc1-lgbm_cfo_large-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d kc1  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-lgbm_cfo_large-Hype-0 2>./stdout/err_kc1-lgbm_cfo_large-Hype-0"
sleep 10s
screen -Sdm kc1-lgbm_cfo_large-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d kc1  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-lgbm_cfo_large-CFO-0 2>./stdout/err_kc1-lgbm_cfo_large-CFO-0"
sleep 10s
screen -Sdm kc1-lgbm_cfo_large-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d kc1  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-lgbm_cfo_large-BS+O-0 2>./stdout/err_kc1-lgbm_cfo_large-BS+O-0"
sleep 10s
screen -Sdm kc1-lgbm_cfo_large-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d kc1  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_kc1-lgbm_cfo_large-Optu-0 2>./stdout/err_kc1-lgbm_cfo_large-Optu-0"
sleep 10s
screen -Sdm phon-lgbm_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d phoneme  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-lgbm_cfo-Neve-0 2>./stdout/err_phon-lgbm_cfo-Neve-0"
sleep 10s
screen -Sdm phon-lgbm_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d phoneme  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-lgbm_cfo-Ax-0 2>./stdout/err_phon-lgbm_cfo-Ax-0"
sleep 10s
screen -Sdm phon-lgbm_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d phoneme  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-lgbm_cfo-Hype-0 2>./stdout/err_phon-lgbm_cfo-Hype-0"
sleep 10s
screen -Sdm phon-lgbm_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d phoneme  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-lgbm_cfo-CFO-0 2>./stdout/err_phon-lgbm_cfo-CFO-0"
sleep 10s
screen -Sdm phon-lgbm_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d phoneme  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-lgbm_cfo-BS+O-0 2>./stdout/err_phon-lgbm_cfo-BS+O-0"
sleep 10s
screen -Sdm phon-lgbm_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d phoneme  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-lgbm_cfo-Optu-0 2>./stdout/err_phon-lgbm_cfo-Optu-0"
sleep 10s
screen -Sdm phon-lgbm_cfo_large-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d phoneme  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-lgbm_cfo_large-Neve-0 2>./stdout/err_phon-lgbm_cfo_large-Neve-0"
sleep 10s
screen -Sdm phon-lgbm_cfo_large-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d phoneme  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-lgbm_cfo_large-Ax-0 2>./stdout/err_phon-lgbm_cfo_large-Ax-0"
sleep 10s
screen -Sdm phon-lgbm_cfo_large-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d phoneme  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-lgbm_cfo_large-Hype-0 2>./stdout/err_phon-lgbm_cfo_large-Hype-0"
sleep 10s
screen -Sdm phon-lgbm_cfo_large-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d phoneme  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-lgbm_cfo_large-CFO-0 2>./stdout/err_phon-lgbm_cfo_large-CFO-0"
sleep 10s
screen -Sdm phon-lgbm_cfo_large-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d phoneme  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-lgbm_cfo_large-BS+O-0 2>./stdout/err_phon-lgbm_cfo_large-BS+O-0"
sleep 10s
screen -Sdm phon-lgbm_cfo_large-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d phoneme  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_phon-lgbm_cfo_large-Optu-0 2>./stdout/err_phon-lgbm_cfo_large-Optu-0"
sleep 10s
screen -Sdm segm-lgbm_cfo-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d segment  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-lgbm_cfo-Neve-0 2>./stdout/err_segm-lgbm_cfo-Neve-0"
sleep 10s
screen -Sdm segm-lgbm_cfo-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d segment  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-lgbm_cfo-Ax-0 2>./stdout/err_segm-lgbm_cfo-Ax-0"
sleep 10s
screen -Sdm segm-lgbm_cfo-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d segment  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-lgbm_cfo-Hype-0 2>./stdout/err_segm-lgbm_cfo-Hype-0"
sleep 10s
screen -Sdm segm-lgbm_cfo-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d segment  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-lgbm_cfo-CFO-0 2>./stdout/err_segm-lgbm_cfo-CFO-0"
sleep 10s
screen -Sdm segm-lgbm_cfo-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d segment  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-lgbm_cfo-BS+O-0 2>./stdout/err_segm-lgbm_cfo-BS+O-0"
sleep 10s
screen -Sdm segm-lgbm_cfo-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo -d segment  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-lgbm_cfo-Optu-0 2>./stdout/err_segm-lgbm_cfo-Optu-0"
sleep 10s
screen -Sdm segm-lgbm_cfo_large-Neve-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d segment  -m Nevergrad -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-lgbm_cfo_large-Neve-0 2>./stdout/err_segm-lgbm_cfo_large-Neve-0"
sleep 10s
screen -Sdm segm-lgbm_cfo_large-Ax-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d segment  -m Ax -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-lgbm_cfo_large-Ax-0 2>./stdout/err_segm-lgbm_cfo_large-Ax-0"
sleep 10s
screen -Sdm segm-lgbm_cfo_large-Hype-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d segment  -m HyperOpt -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-lgbm_cfo_large-Hype-0 2>./stdout/err_segm-lgbm_cfo_large-Hype-0"
sleep 10s
screen -Sdm segm-lgbm_cfo_large-CFO-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d segment  -m CFO -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-lgbm_cfo_large-CFO-0 2>./stdout/err_segm-lgbm_cfo_large-CFO-0"
sleep 10s
screen -Sdm segm-lgbm_cfo_large-BS+O-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d segment  -m BlendSearch+Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-lgbm_cfo_large-BS+O-0 2>./stdout/err_segm-lgbm_cfo_large-BS+O-0"
sleep 10s
screen -Sdm segm-lgbm_cfo_large-Optu-0 bash -c "python test/test_automl_exp.py  -t 14400.0 -l lgbm_cfo_large -d segment  -m Optuna -t 14400.0 -total_pu 1 -trial_pu 1  -r 0 >./stdout/out_segm-lgbm_cfo_large-Optu-0 2>./stdout/err_segm-lgbm_cfo_large-Optu-0"
sleep 10s