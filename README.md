# Talaba-balallariimport numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


np.random.seed(42)
x = np.random.uniform(2, 5, 30)  # O'zgaruvchi x
y = 35 + 5 * x + np.random.normal(0, 2, len(x))

# DataFrame yaratish
df = pd.DataFrame({'x': x, 'y': y})

# Chiziqli regressiya modelini qurish
X = sm.add_constant(df['x'])
model = sm.OLS(df['y'], X).fit()
df['y_pred'] = model.predict(X)

# Grafik chizish
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.regplot(x='x', y='y_pred', data=df, scatter=False, ax=axes[0], color='blue', ci=None, line_kws={'linestyle':'dashed'})
axes[0].set_title(r'$y = mx + c$')

sns.regplot(x='x', y='y', data=df, ax=axes[1], color='black', line_kws={'color':'blue', 'linestyle':'dashed'})
axes[1].set_title(r"$y' = b0 + b1x$")
axes[1].set_ylabel("mpg")

plt.tight_layout()
plt.show()
