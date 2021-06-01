# Environment requirements


# Commmand line to run exp
## Rerun an exp
python test/vw/run_autovw.py  final_icml 100k_1_5 -m ChaCha  Vanilla   -seed 0  -rerun -log -d oml_41506_10
## Plot results 
python test/vw/run_autovw.py  final_icml 100k_1_5 -m ChaCha  Vanilla   -seed 0  -p -log -d oml_41506_10
python test/vw/run_autovw.py  final_icml 100k_1_5 -m ChaCha  Vanilla   -seed 0  -p  -log -d oml_42183_10