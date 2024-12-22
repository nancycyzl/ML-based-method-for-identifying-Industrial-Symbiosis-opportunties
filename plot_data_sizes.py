import matplotlib.pyplot as plt
import os

produce_aucs = [0.9293, 0.9194, 0.9180, 0.9164]
need_aucs = [0.9061, 0.8999, 0.9014, 0.8964]

x_values = ["Full", "90%", "80%", "70%"]
font_size = 18

plt.figure(figsize=(8, 8))
plt.plot(x_values, produce_aucs, color="tab:blue", marker='o', markersize=10, linewidth=2, label="Produce Model")
plt.plot(x_values, need_aucs, color="tab:orange", marker='o', markersize=10, linewidth=2, label="Need Model")
plt.xlabel('Data size', fontsize=font_size)
plt.ylabel('AUC', fontsize=font_size)
plt.ylim([0.875, 0.95])
plt.xticks(fontsize=font_size)
plt.yticks([0.875, 0.9, 0.925, 0.95], fontsize=font_size)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=font_size)
plt.tight_layout()
plt.savefig("output_new/SA_data_size.png")
