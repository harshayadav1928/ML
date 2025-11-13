from scipy import stats
import matplotlib.pyplot as plt



group1 = [12,14,15,10,13]
group2 = [8,9,12,11,7]

t_stat, p_val = stats.ttest_ind(group1, group2)
print("t-statistic:", t_stat)
print("p-value:", p_val)
plt.bar(['Group 1', 'Group 2'], [sum(group1)/len(group1), sum(group2)/len(group2)])
plt.ylabel('Mean Value')
plt.title('Mean Comparison (t-test)')
plt.show()