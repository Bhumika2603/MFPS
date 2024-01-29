
def write_report(report_dict,algo_path):
    algo_path = algo_path+'/result.txt'
    with open(algo_path,"w") as file1:
        for k,v in report_dict.items():
            file1.write("{} : {} \n".format(k,v))
    file1.close()

def storing_model(base_dir,task_,algo):
    import os

    task_path = base_dir+task_
    if not os.path.exists(task_path):
        os.mkdir(task_path)

    algo_path = task_path +algo
    if not os.path.exists(algo_path):
        os.mkdir(algo_path)

    return algo_path
    
