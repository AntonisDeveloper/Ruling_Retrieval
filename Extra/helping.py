import pandas as pd

df = pd.read_csv("swiss_rulings_with_article_references.csv")

num_of_rulings = len(df)
print(f"Number of rulings: {num_of_rulings}")

small_df = df.drop(columns=[x for x in df.columns if x not in ["decision_id", "full_text_en"]])

small_df.to_csv("swiss_rulings_with_article_references_small.csv", index=False)


