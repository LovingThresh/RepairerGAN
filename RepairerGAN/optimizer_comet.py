# -*- coding: utf-8 -*-
# @Time    : 2022/4/26 22:58
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : optimizer_comet.py
# @Software: PyCharm

from comet_ml import Optimizer

# 设定超参、模型
config = {
    "algorithm": "bayes",
    "parameters": {
        "x": {"type": "integer", "min": 1, "max": 5}},
    "spec": {
        "metric": "loss",
        "objective": "minimize"
    }
}

opt = Optimizer(config)

for experiment in opt.get_experiments(project_name="optimizer-search-01"):
    print(experiment)
