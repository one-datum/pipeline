# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

plt.plot([0, 1], [0, 1], "k")
plt.xlabel("face")

plt.savefig("demo_figure.pdf", bbox_inches="tight")
