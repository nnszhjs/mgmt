import pandas as pd
# 1. 读取原始数据
df = pd.read_csv('/rmbs_1/RecBole/dataset/movielen/ratings.csv', sep=',')
# 2. 统一列名为小写（RecBole 默认习惯）
# df = df[["userid","itemid","rating","time"]]
df.columns = ['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float']
# 3. 写入 .inter 文件
output_path = '/rmbs_1/RecBole/dataset/movielen/movielen.inter'
    
# 追加数据（不包含 pandas 默认索引）
df.to_csv(output_path, sep='\t', index=False, mode="w")
print(f"转换完成！文件已保存至 {output_path}")