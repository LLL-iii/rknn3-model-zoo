from vlmeval.evaluate import *

Golden_file = "MMBench_EN_Golden.xlsx"      ## 英文数据集Golden文件
Eval_file = "results_en.jsonl"              ## 板端运行得到的英文数据集结果文件

acc = multiple_choice_eval(Golden_file, Eval_file, **{'nproc': 1, 'verbose': True, 'model': 'chatgpt-0613'})

print("The acc is :", acc)