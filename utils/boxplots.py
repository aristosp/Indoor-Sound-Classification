import pandas as pd
import matplotlib.pyplot as plt

same_kernels = pd.read_csv('../2d_convs/logs_same_kernel.csv')
diff_kernels = pd.read_csv('../2d_convs/logs_diffkernels.csv')
vit = pd.read_csv('../vision_transformers/logs_transformer.csv')
adam_1d = pd.read_csv('../1d_convs/conv1d_logs_adam.csv')
# adabound_1d = pd.read_csv('../1d_convs/conv1d_logs_adabound.csv')
f1scores_same = same_kernels['Macro-F1 score']
f1scores_diff = diff_kernels['Macro-F1 score']
f1scores_vit = vit['Macro-F1 score']
f1scores_1d_adam = adam_1d['Macro-F1 score']
# f1scores_1d = adabound_1d['Macro-F1 score']
f1scores = [f1scores_1d_adam, f1scores_same, f1scores_diff, f1scores_vit]
fig, ax = plt.subplots(figsize=(10, 7))
plt.boxplot(f1scores)
plt.title('Box Plots for F1 score for all models')
plt.ylabel('F1 Score')
plt.xlabel('Models')
ax.set_xticklabels(['Conv1D', 'Conv2D,with same kernels',
                    'Conv2D, with different kernels', 'Vision Transformer'])
plt.show()