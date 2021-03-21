
## all exp 4 methods
# python gene_script.py  -p automl -l all -t 3600.0  -m BlendSearch-FLOW2-OptunaBO-update OptunaBO MultiStartLS-FLOW2-rsg-1 BOwLS-FLOW2-OptunaBO -rerun  -evaluate -filename all.sh
# screen -Sdm adult-xgb-Optuna-4 python test/test_automl_exp.py  -t 600 -trial_pu 1  -total_pu 1  -m 'Optuna' -l xgb_cat -d adult -r 5
# python gene_script.py  -l xgb_cat -t 600.0  -m Optuna CFO BlendSearch+Optuna -plot_only -agg  -f 0 -filename xgb.sh

### Example command for Windows machine
# python gene_script.py  -l xgb_cat -t 60.0  -m Optuna CFO BlendSearch+Optuna -d vehicle  -f 5 -filename xgb_cat.sh -export
# python gene_script.py  -l xgb_cat -t 60.0  -m Optuna CFO BlendSearch+Optuna -d vehicle  -f 5 -filename xgb_cat.sh 
# python gene_script.py  -l xgb_cat -t 60.0  -m Optuna CFO BlendSearch+Optuna -d vehicle -filename xgb.sh -plot_only -agg 

### Command to run all datasets
# python gene_script.py  -l xgb_cat -t 60.0  -m Optuna CFO BlendSearch+Optuna -filename xgb_cat_all.sh -f 0 1 2 3 4 
# python gene_script.py  -l xgb_cat -t 60.0  -m Optuna CFO BlendSearch+Optuna -filename xgb_cat_all.sh -agg -plot_only
import argparse

if __name__=='__main__':
    ###************************Datasets for ICLR21**************************************
    ###====================xgb_cat/lgbm datasets (34 + 5): =========================
    # datasetlist 1h
    dataset_small = ['blood', 'Australian', 'credit', 'car','kc1',  'kr', 'phoneme',  'segment', ]
    dataset_large_1= [
        'Airlines', 'christine', 'shuttle',  'connect', 'sylvine',
        'guillermo', 'volkert', 'MiniBooNE', 'Jannis', 'mfeat', 
        'jungle', 'jasmine','riccardo', 'higgs', 'fabert', 'cnae', 
        ] #'Fashion',
    dataset_large_2= [
        'bank', 'Albert',  'APSFailure',  'nomao', 'numerai28', 
        'Helena','KDDCup09', 'adult', 'Amazon', 'vehicle', 'Dionis', 'dilbert','Covertype','Robert', 'Fashion'
        ] #'Fashion',
    dataset_list_1h = [
        'Airlines', 'christine', 'shuttle', 'car', 'credit', 'connect', 'sylvine',
        'guillermo', 'volkert', 'MiniBooNE', 'Jannis', 'mfeat', 'jungle', 'jasmine',
        'riccardo', 'higgs', 'fabert', 'segment', 'cnae', 'kc1', 'kr',
        'Albert',  'APSFailure', 'bank', 'nomao', 'numerai28', 'phoneme',
        'Helena','KDDCup09', 'adult', 'Amazon', 'vehicle', 'blood', 'Australian',
        ] #'Fashion',
    # datasetlist 4h
    dataset_list_4h = ['Dionis', 'dilbert','Covertype','Robert', 'Fashion'] 
    dataset_list_iclr = dataset_list_1h + dataset_list_4h
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--time', metavar='time', type = float, 
        default=1, help="time_budget")
    parser.add_argument('-i', '--iter', metavar='iter', type = int, 
        default=0, help="iter_budget")
    parser.add_argument('-f', '--fold_num', dest='fold_num', nargs='*' , 
        default= [], help="The fold number")
    parser.add_argument('-f_each_run_num', '--fold_num_each_run', metavar='fold_num_each_run', type = int, 
        default=1, help="number of folds in each run")
    parser.add_argument('-c', '--core_num', metavar='core_num', type = int, 
        default=1, help="core num")
    parser.add_argument('-c_trial', '--core_per_trial', metavar='core_per_trial', type = int, 
        default=1, help="core_per_trial")
    parser.add_argument('-m', '--method_list', dest='method_list', nargs='*' , 
        default= [], help="The method list")
    parser.add_argument('-l', '--learner_list', dest='learner_list', nargs='*' , 
        default= ['all'], help="The learn list")
    parser.add_argument('-d', '--dataset_list', dest='dataset_list', nargs='*' , 
        default= [], help="The dataset list") # ['cnae','shuttle', ] 
    # parser.add_argument('-rerun', '--force_rerun', action='store_true',
    #                     help='whether to force rerun.') 
    parser.add_argument('-plot_only', '--plot_only', action='store_true',
                        help='whether to plot only.')
    parser.add_argument('-agg', '--agg', action='store_true',
                        help='whether to aggregrate results.')
    parser.add_argument('-no_redirect', '--no_redirect', action='store_true',
                        help='whether to redirect std output.')
    parser.add_argument('-export', '--export', action='store_true',
                        help='whether to redirect std output.')
    parser.add_argument('-filename', '--file_name', metavar='file_name',  
        default='run_exp.sh', help="result filename")
    parser.add_argument('-max_job', '--max_job', metavar='max_job', type = int, 
        default=25, help="max number of jobs to include in a file")
    parser.add_argument('-dlist', '--dlist', metavar='dlist', type = str, 
        default=None, help="data list")
    args = parser.parse_args()
    max_job = args.max_job
    time_budget = args.time # seconds
    # iter_budget = args.iter # iterations, for synthetic study only
    learner_list = args.learner_list 
    method_list = args.method_list
    alias_time = '-t ' + str(time_budget)
    alias_core = '-total_pu ' + str(args.core_num)
    alias_core_per_trial = '-trial_pu ' + str(args.core_per_trial)
    additional_argument = ''
    if args.plot_only:
        additional_argument += ' -plot_only '
    if args.agg:
        additional_argument += ' -agg '
    
    argument_list = []
    if args.dlist:
        if 'small' in args.dlist: dataset_list = dataset_small
        elif 'large1' in args.dlist: dataset_list = dataset_large_1 
        elif 'large2' in args.dlist: dataset_list = dataset_large_2 
        elif 'all' in args.dlist or 'iclr' in args.dlist: dataset_list = dataset_list_iclr 
    else:
        dataset_list = args.dataset_list
        if len(dataset_list) == 0:
            dataset_list = dataset_list_iclr
    # filename = 'run_flaml_bs.sh'
    filename = args.file_name
    f = open(filename,'w')
    if args.export:
        f.write('export SCREENDIR=$HOME/.screen')
    fold_list = args.fold_num
    frun_num = args.fold_num_each_run
    no_redirect = args.no_redirect
    if len(fold_list)!=0: 
        fold_chunk_list = [fold_list[i * frun_num:(i + 1) * frun_num
            ] for i in range((len(fold_list) + frun_num - 1) // frun_num )] 
    job_counter = 0
    for d in dataset_list:
        for l in learner_list:
            argu_learner = '-l ' + str(l)
            argu_d = '-d ' + str(d)
            alias_screen =  d[0:4] + '-' + str(l)
            if args.agg: 
                method_list =  [ ' '.join([m for m in args.method_list])]
                alias_screen = 'agg-' + alias_screen
            for m in method_list:
                method_short = m.replace('BlendSearch', 'BS').replace('FLOW2', 'F2')
                alias_screen_method = alias_screen + '-' + method_short[0:4] 
                alias_method = ' -m ' + m
                if len(fold_list)==0:
                    argument_list = [alias_time, argu_learner, argu_d, alias_method, alias_core, alias_core_per_trial,
                        additional_argument]
                    line_part1 = '\n'+ 'screen -Sdm ' + alias_screen_method + ' ' + 'bash -c '
                    line_part2 = '"' + 'python test/test_automl_exp.py  ' + ' '.join(argument_list) \
                                + '>./stdout/out_' + alias_screen_method \
                                + ' ' + '2>./stdout/err_' + alias_screen_method + '"'
                    if not no_redirect:
                        line = line_part1 + line_part2 
                    else:
                        line = '\n'+ 'screen -Sdm ' + alias_screen_method \
                            + ' ' + ' python test/test_automl_exp.py  ' + ' '.join(argument_list) 
                    f.write(line) 
                    f.write('\n' + 'sleep 10s')
                    job_counter +=1
                    if job_counter%max_job==0:
                            f.write('\n' + 'sleep ' + str(int(args.time*1.1)) + 's')
                else:
                    for f_chunk in fold_chunk_list:
                        alias_fold = ' -r ' + ' '.join(f_chunk) 
                        alias_screen_method_fold = alias_screen_method + '-' + f_chunk[0] 
                        argument_list = [alias_time, argu_learner, argu_d, alias_method, alias_time, alias_core, alias_core_per_trial,
                           alias_fold, additional_argument]
                        line_part1 = '\n'+ 'screen -Sdm ' \
                            + alias_screen_method_fold + ' ' + 'bash -c '
                        line_part2 = '"' + 'python test/test_automl_exp.py  ' \
                            + ' '.join(argument_list) \
                            + '>./stdout/out_' + alias_screen_method_fold \
                            + ' ' + '2>./stdout/err_' + alias_screen_method_fold + '"'
                        if not no_redirect:
                            line = line_part1 + line_part2 
                        else:
                            line = '\n' + 'screen -Sdm ' \
                                + alias_screen_method_fold + 'python test/test_automl_exp.py  ' \
                                + ' '.join(argument_list) 
                        f.write(line)
                        f.write('\n' + 'sleep 10s')
                        job_counter +=1
                        if job_counter%max_job==0:
                            f.write('\n' + 'sleep ' + str(int(args.time*1.1)) + 's')
        