import matplotlib.pyplot as plt
import numpy as np


def plot_training_loss(loss_history, output_path='loss_plot.png'):
    iterations = np.arange(1, len(loss_history) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, loss_history, linewidth=2, color='blue')
    plt.xlabel('iteration_number', fontsize=12)
    plt.ylabel('training_loss', fontsize=12)
    plt.title('BCE loss plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Loss plot saved to {output_path}")
    plt.close()
