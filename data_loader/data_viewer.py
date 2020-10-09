#encoding=utf-8
import pandas as pd

train_set_path = "../data/census_data/adult.data"
test_set_path = "../data/census_data/adult.test"

train_set = pd.read_csv(train_set_path)
fnlwgt_col = train_set.iloc[:, 2] / 100
print(fnlwgt_col.describe())
sections = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400,
            2600, 2800, 3000, 3500, 4000, 4500, 5000, 20000]
group_names = list(map(str, sections))[:-1]
cuts = pd.cut(fnlwgt_col, sections, labels=group_names)
counts_dict = dict(pd.value_counts(cuts))
sorted_list = sorted(counts_dict.items(), key=lambda x: int(x[0]))
for ele in sorted_list:
    print(ele)