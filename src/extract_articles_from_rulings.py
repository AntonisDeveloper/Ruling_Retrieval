import re
import pandas as pd
from collections import defaultdict

pattern = r'''
\b
(?:art\.?|artikel)?\s*                            # art., Art., artikel (optional)
(?P<article>\d+[a-zA-Z]?)\s*                      # article number (e.g. 95 or 95a)

(?:                                               # optional paragraph/Abs./para
    (?:abs\.?|para\.?|paragraph|al\.?)\s*
    (?P<para>\d+[a-zA-Z]?)\s*
)? 

(?:                                               # optional lit./let./bst
    (?:lit\.?|let\.?|bst\.?)\s*
    (?P<lit>[a-z])\s*                            # single letter only
)? 

(?P<law>ZGB|OR|StGB|BGG|BVG|BV|ZPO|StPO|IVG|OJ|of the Criminal Code)  # specific Swiss law codes
\b
'''
df = pd.read_csv("swiss_rulings_with_translations.csv")

# Initialize new columns
df['articles_mentioned'] = None
df['full_article_references'] = None

# Initialize counters and statistics
total_matches = 0
matches_per_ruling = defaultdict(int)
law_stats = defaultdict(int)
article_stats = defaultdict(int)

for index, row in df.iterrows():
    text = row["full_text_en"]
    dec_id = row["decision_id"]
    matches = re.finditer(pattern, text, re.VERBOSE | re.IGNORECASE)
    
    ruling_matches = 0
    articles_mentioned = []
    full_references = []
    
    print("\nDecision ID: ", dec_id)
    for m in matches:
        ruling_matches += 1
        match_dict = m.groupdict()
        print(match_dict)
        
        # Create the simple article reference
        if match_dict['article'] and match_dict['law']:
            simple_ref = f"Art. {match_dict['article']} {match_dict['law']}"
            articles_mentioned.append(simple_ref)
            
            # Create the full reference
            full_ref = f"Art. {match_dict['article']}"
            if match_dict['para']:
                full_ref += f" para. {match_dict['para']}"
            if match_dict['lit']:
                full_ref += f" lit. {match_dict['lit']}"
            full_ref += f" {match_dict['law']}"
            full_references.append(full_ref)
        
        # Update statistics
        if match_dict['law']:
            law_stats[match_dict['law']] += 1
        if match_dict['article'] and match_dict['law']:
            article_key = f"Article {match_dict['article']} of {match_dict['law']}"
            article_stats[article_key] += 1
    
    # Update DataFrame with the lists
    df.at[index, 'articles_mentioned'] = articles_mentioned
    df.at[index, 'full_article_references'] = full_references
    
    matches_per_ruling[dec_id] = ruling_matches
    total_matches += ruling_matches

# Print overall statistics
print("\n=== Overall Statistics ===")
print(f"Total number of matches found: {total_matches}")
print(f"Number of rulings with matches: {len(matches_per_ruling)}")
print(f"Average matches per ruling: {total_matches/len(df):.2f}")

# Print top 10 most referenced laws
print("\n=== Top 10 Most Referenced Laws ===")
for law, count in sorted(law_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{law}: {count} references")

# Print top 10 most referenced articles
print("\n=== Top 10 Most Referenced Articles ===")
for article_ref, count in sorted(article_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{article_ref}: {count} references")

# Print rulings with most matches
print("\n=== Top 10 Rulings with Most Matches ===")
for dec_id, count in sorted(matches_per_ruling.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"Decision {dec_id}: {count} matches")

# Save the updated DataFrame
print("\nSaving updated DataFrame...")
df.to_csv("swiss_rulings_with_article_references.csv", index=False)
print("Done! Saved to swiss_rulings_with_article_references.csv")

