import matplotlib.pyplot as plt
import numpy as np
import os


# Data
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
our_PR = [0.8652, 0.8772, 0.8671, 0.8721]
our_NR = [0.8232, 0.8049, 0.8495, 0.8266]
llama_PR = [0.6845, 0.8535, 0.4884, 0.6213]   # llama 3.1 - 8B
llama_NR = [0.6159, 0.7615, 0.3287, 0.4592]
gpt_PR = [0.6371, 0.9823, 0.3208, 0.4837]  # gpt-4o-mini
gpt_NR = [0.5295, 0.8611, 0.0614, 0.1146]

# Plot settings
x = np.arange(len(metrics))
bar_width = 0.2
fontsize = 18


# Plot for PR
plt.figure(figsize=(10, 6))
plt.bar(x - bar_width, llama_PR, width=bar_width, label='Llama3.1', color='#66b0e5', alpha=1)
plt.bar(x, gpt_PR, width=bar_width, label='GPT-4o-mini', color='#1f77b4', alpha=1)
plt.bar(x + bar_width, our_PR, width=bar_width, label='Proposed model', color='#165683', alpha=1)

plt.xticks(x, metrics, fontsize=fontsize)
plt.yticks(fontsize=fontsize)
# plt.xlabel("Metrics", fontsize=fontsize)
plt.ylabel("Scores", fontsize=fontsize)
plt.title("Performance Metrics for Produce Relationship (PR)", fontsize=fontsize)
# plt.legend(fontsize=fontsize-2)
plt.tight_layout()
plt.savefig("output_LLM/PR_Performance.png")


# Plot for NR
plt.figure(figsize=(10, 6))
plt.bar(x - bar_width, llama_NR, width=bar_width, label='Llama3.1', color='#66b0e5', alpha=1)
plt.bar(x, gpt_NR, width=bar_width, label='GPT-4o-mini', color='#1f77b4', alpha=1)
plt.bar(x + bar_width, our_NR, width=bar_width, label='Proposed model', color='#165683', alpha=1)

plt.xticks(x, metrics, fontsize=fontsize-2)
plt.yticks(fontsize=fontsize)
# plt.xlabel("Metrics", fontsize=fontsize)
plt.ylabel("Scores", fontsize=fontsize)
plt.title("Performance Metrics for Need Relationship (NR)", fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.tight_layout()
plt.savefig("output_LLM/NR_Performance.png")

