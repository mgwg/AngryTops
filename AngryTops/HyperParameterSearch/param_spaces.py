"""The HyperParameter spaces. Each space corresponds to a different model"""
from hyperopt import hp

space0 = {
'learn_rate': hp.uniform('learn_rate', 10e-6, 10e-4),
'size1': hp.quniform('size1', 1, 200, 1),
'size2': hp.quniform('size2', 1, 200, 1),
'size3': hp.quniform('size3', 1, 200, 1),
'size4': hp.quniform('size4', 1, 200, 1),
'size5': hp.quniform('size5', 1, 200, 1),
'size6': hp.quniform('size6', 1, 200, 1),
'size7': hp.quniform('size7', 1, 200, 1),
'act1': hp.choice('act1', ['relu', 'elu', 'tanh']),
'act2': hp.choice('act2', ['relu', 'elu', 'tanh']),
'act3': hp.choice('act3', ['relu', 'elu', 'tanh']),
'act4': hp.choice('act4', ['relu', 'elu', 'tanh']),
'reg_weight': hp.uniform('reg_weight', 0, 1),
'rec_weight': hp.uniform('rec_weight', 0, 1)
}

space1 = {
'learn_rate': hp.uniform('learn_rate', 10e-6, 10e-4),
'size1': hp.quniform('size1', 1, 200, 1),
'size2': hp.quniform('size2', 1, 200, 1),
'size3': hp.quniform('size3', 1, 200, 1),
'size4': hp.quniform('size4', 1, 200, 1),
'size5': hp.quniform('size5', 1, 200, 1),
'act1': hp.choice('act1', ['relu', 'elu', 'tanh']),
'act2': hp.choice('act2', ['relu', 'elu', 'tanh']),
'act3': hp.choice('act3', ['relu', 'elu', 'tanh']),
'act4': hp.choice('act4', ['relu', 'elu', 'tanh']),
'act5': hp.choice('act5', ['relu', 'elu', 'tanh']),
'reg_weight': hp.uniform('reg_weight', 0, 1),
'rec_weight': hp.uniform('rec_weight', 0, 1)
}

space2 = {
'learn_rate': hp.uniform('learn_rate', 10e-6, 10e-4),
'size1': hp.quniform('size1', 1, 200, 1),
'size2': hp.quniform('size2', 1, 200, 1),
'size3': hp.quniform('size3', 1, 200, 1),
'size4': hp.quniform('size4', 1, 200, 1),
'size5': hp.quniform('size5', 1, 200, 1),
'size6': hp.quniform('size6', 1, 200, 1),
'size7': hp.quniform('size7', 1, 200, 1),
'size8': hp.quniform('size8', 1, 200, 1),
'act1': hp.choice('act1', ['relu', 'elu', 'tanh']),
'act2': hp.choice('act2', ['relu', 'elu', 'tanh']),
'act3': hp.choice('act3', ['relu', 'elu', 'tanh']),
'act4': hp.choice('act4', ['relu', 'elu', 'tanh']),
'act5': hp.choice('act5', ['relu', 'elu', 'tanh']),
'act6': hp.choice('act6', ['relu', 'elu', 'tanh']),
'kernel_reg1': hp.uniform('kernel_reg1', 0, 1),
'kernel_reg2': hp.uniform('kernel_reg2', 0, 1),
'kernel_reg3': hp.uniform('kernel_reg3', 0, 1),
'kernel_reg4': hp.uniform('kernel_reg4', 0, 1),
'kernel_reg5': hp.uniform('kernel_reg5', 0, 1)
}

space3 = {
'learn_rate': hp.uniform('learn_rate', 10e-6, 10e-4),
'size1': hp.quniform('size1', 1, 200, 1),
'size2': hp.quniform('size2', 1, 200, 1),
'size3': hp.quniform('size3', 1, 200, 1),
'size4': hp.quniform('size4', 1, 200, 1),
'size5': hp.quniform('size5', 1, 200, 1),
'size6': hp.quniform('size6', 1, 200, 1),
'size7': hp.quniform('size7', 1, 200, 1),
'act1': hp.choice('act1', ['relu', 'elu', 'tanh']),
'act2': hp.choice('act2', ['relu', 'elu', 'tanh']),
'act3': hp.choice('act3', ['relu', 'elu', 'tanh']),
'act4': hp.choice('act4', ['relu', 'elu', 'tanh']),
'act5': hp.choice('act5', ['relu', 'elu', 'tanh']),
'kernel_reg1': hp.uniform('kernel_reg1', 0, 1),
'kernel_reg2': hp.uniform('kernel_reg2', 0, 1),
'kernel_reg3': hp.uniform('kernel_reg3', 0, 1),
'kernel_reg4': hp.uniform('kernel_reg4', 0, 1),
'kernel_reg5': hp.uniform('kernel_reg5', 0, 1)
}

space4 = {
'learn_rate': hp.uniform('learn_rate', 10e-6, 10e-4),
'size1': hp.quniform('size1', 1, 200, 1),
'size2': hp.quniform('size2', 1, 200, 1),
'size3': hp.quniform('size3', 1, 200, 1),
'size4': hp.quniform('size4', 1, 200, 1),
'size5': hp.quniform('size5', 1, 200, 1)
}

space4_1 = {
'learn_rate': hp.uniform('learn_rate', 10e-6, 10e-4),
'size1': hp.quniform('size1', 1, 1000, 1),
'size2': hp.quniform('size2', 1, 1000, 1),
'size3': hp.quniform('size3', 1, 1000, 1),
'size4': hp.quniform('size4', 1, 1000, 1),
'size5': hp.quniform('size5', 1, 1000, 1)
}

cnn_space1 = {
'learn_rate': hp.uniform('learn_rate', 10e-6, 10e-3),
'size1': hp.quniform('size1', 1, 200, 1),
'size2': hp.quniform('size2', 1, 200, 1),
'size3': hp.quniform('size3', 1, 200, 1),
'act1': hp.choice('act1', ['relu', 'elu', 'tanh']),
'act2': hp.choice('act2', ['relu', 'elu', 'tanh']),
'act3': hp.choice('act3', ['relu', 'elu', 'tanh'])
}

cnn_space2 = {
'learn_rate': hp.uniform('learn_rate', 10e-6, 10e-3),
'size1': hp.quniform('size1', 1, 200, 1),
'size2': hp.quniform('size2', 1, 200, 1),
'size3': hp.quniform('size3', 1, 200, 1),
'size4': hp.quniform('size4', 1, 200, 1),
'act1': hp.choice('act1', ['relu', 'elu', 'tanh']),
'act2': hp.choice('act2', ['relu', 'elu', 'tanh']),
'act3': hp.choice('act3', ['relu', 'elu', 'tanh']),
'act4': hp.choice('act4', ['relu', 'elu', 'tanh'])
}

parameter_spaces = {
'space0': space0, 'space1': space1, 'space2': space2, 'space3': space3,
'cnn_space1': cnn_space1, 'cnn_space2': cnn_space2, 'space4': space4,
'space4_1': space4_1
}
