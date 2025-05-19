import pandas as pd

def extract_first_three_columns(input_csv, output_csv):
    """
    从输入的 CSV 文件中提取前三列，并将这些列保存到新的 CSV 文件中。

    :param input_csv: 输入的 CSV 文件路径
    :param output_csv: 输出的 CSV 文件路径
    """
    # 读取输入的 CSV 文件
    df = pd.read_csv(input_csv)
    
    # 提取前三列
    first_three_columns = df.iloc[:, :3]
    
    # 将提取的列保存到新的 CSV 文件中
    first_three_columns.to_csv(output_csv, index=False)

# 示例用法
input_csv_path = './to_submit.csv'
output_csv_path = './output-3-cols.csv'
extract_first_three_columns(input_csv_path, output_csv_path)