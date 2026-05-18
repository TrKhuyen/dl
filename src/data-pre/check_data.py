# import os
# from datasets import load_from_disk

# metadata = load_from_disk(r"D:\dl\dl\data\raw\vietnamese_legal_metadata")
# content = load_from_disk(r"D:\dl\dl\data\raw\vietnamese_legal_content")

# metadata_df = metadata["data"].to_pandas()
# content_df = content["data"].to_pandas()

# df = content_df.merge(metadata_df, on="id", how="left")
# # print(df)

# target_id = 680604
# record = df.loc[df["id"] == target_id, ["id", "content"]]

# if record.empty:
# 	print(f"\nKhong tim thay ban ghi co id = {target_id}")
# else:
# 	print(f"\nContent cua id {target_id}:\n")
# 	print(record.iloc[0]["content"])
# record.to_csv(r"D:\dl\dl\data\raw\record_680604.csv", index=False)
# # df.to_csv(r"D:\dl\dl\data\raw\vietnamese_legal_data.csv", index=False)


# check metadata & content
from datasets import load_from_disk

metadata = load_from_disk(r"D:\dl\dl\data\raw\vietnamese_legal_metadata")
content = load_from_disk(r"D:\dl\dl\data\raw\vietnamese_legal_content")

metadata_df = metadata["data"].to_pandas()
content_df = content["data"].to_pandas()

print("Metadata shape:", metadata_df.shape)
print("Content shape:", content_df.shape)

print("Metadata columns:", metadata_df.columns.tolist())
print("Content columns:", content_df.columns.tolist())

print("Duplicate id metadata:", metadata_df["id"].duplicated().sum())
print("Duplicate id content:", content_df["id"].duplicated().sum())

df = content_df.merge(metadata_df, on="id", how="left")

print("Merged shape:", df.shape)
print("Missing metadata rows:", df["document_number"].isna().sum())

print(df[["id", "document_number", "title", "issuance_date"]].head())

# import pandas as pd

# df = pd.read_csv(r"D:\dl\dl\data\processed\legal_documents_clean_2024_2026.csv")
# print(df.shape)
# print(df[["id", "document_number", "clean_char_len", "has_articles"]].head())
# print(df["has_articles"].value_counts())