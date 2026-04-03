"""
Simple architecture diagram: 3 boxes - Logit Demand Model, Iterative Base Rules, Q-Table
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
fig.set_facecolor('white')
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis('off')

box_h = 1.8
box_w = 2.5
y_center = 2.0
arrow_color = '#2C3E50'

# Box 1 — Logit Demand Model
x1 = 0.5
b1 = mpatches.FancyBboxPatch((x1, y_center - box_h/2), box_w, box_h,
    boxstyle="round,pad=0.15", facecolor='#E8F4FD', edgecolor='#2980B9', linewidth=2)
ax.add_patch(b1)
ax.text(x1 + box_w/2, y_center + 0.25, 'Logit Demand', fontsize=12, fontweight='bold',
        ha='center', va='center', color='#2980B9')
ax.text(x1 + box_w/2, y_center - 0.25, 'Model', fontsize=12, fontweight='bold',
        ha='center', va='center', color='#2980B9')

# Arrow 1→2
ax.annotate('', xy=(x1 + box_w + 0.6, y_center), xytext=(x1 + box_w + 0.1, y_center),
            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2.5))

# Box 2 — Iterative Base Rules
x2 = 3.75
b2 = mpatches.FancyBboxPatch((x2, y_center - box_h/2), box_w, box_h,
    boxstyle="round,pad=0.15", facecolor='#EBF5EB', edgecolor='#27AE60', linewidth=2)
ax.add_patch(b2)
ax.text(x2 + box_w/2, y_center + 0.25, 'Iterative', fontsize=12, fontweight='bold',
        ha='center', va='center', color='#27AE60')
ax.text(x2 + box_w/2, y_center - 0.25, 'Base Rules', fontsize=12, fontweight='bold',
        ha='center', va='center', color='#27AE60')

# Arrow 2→3
ax.annotate('', xy=(x2 + box_w + 0.6, y_center), xytext=(x2 + box_w + 0.1, y_center),
            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2.5))

# Box 3 — Q-Table
x3 = 7.0
b3 = mpatches.FancyBboxPatch((x3, y_center - box_h/2), box_w, box_h,
    boxstyle="round,pad=0.15", facecolor='#FDF2E9', edgecolor='#E67E22', linewidth=2)
ax.add_patch(b3)
ax.text(x3 + box_w/2, y_center, 'Q-Table', fontsize=13, fontweight='bold',
        ha='center', va='center', color='#E67E22')

plt.tight_layout()
plt.savefig("architecture_diagram.png", dpi=600, bbox_inches='tight', facecolor='white')
print("Saved architecture_diagram.png")