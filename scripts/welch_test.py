from scipy.stats import ttest_ind

image = [
    0.6178250259348985,
    0.6210491126685819,
    0.6176646606140701
]

specimen = [
    0.6466249976694770,
    0.6405540387478669,
    0.6454344667305008
]

t_stat, p_value = ttest_ind(image, specimen, equal_var=False)

print("t =", t_stat)
print("p =", p_value)