from datasets import load_dataset
from deep_translator import GoogleTranslator
import pandas as pd
import time
from tqdm import tqdm
import os
import sys

# List of terms that should not be translated
DO_NOT_TRANSLATE = [
    'Bundesgericht', 'Bundesgerichts',  # Federal Court
    'Kassationsgericht', 'Kassationsgerichts',  # Court of Cassation
    'Obergericht', 'Obergerichts',  # High Court
    'Verwaltungsgericht', 'Verwaltungsgerichts',  # Administrative Court
    'Handelsgericht', 'Handelsgerichts',  # Commercial Court
    'Strafgericht', 'Strafgerichts',  # Criminal Court
    'Zivilgericht', 'Zivilgerichts',  # Civil Court
    'Sozialversicherungsgericht', 'Sozialversicherungsgerichts',  # Social Insurance Court
    'Steuergericht', 'Steuergerichts',  # Tax Court
    'Schlichtungsbehörde', 'Schlichtungsbehörden',  # Conciliation Authority
    'Kantonsgericht', 'Kantonsgerichts',  # Cantonal Court
    'Bezirksgericht', 'Bezirksgerichts',  # District Court
    'Friedensrichter', 'Friedensrichters',  # Justice of the Peace
    'Bundesverwaltungsgericht', 'Bundesverwaltungsgerichts',  # Federal Administrative Court
    'Bundesstrafgericht', 'Bundesstrafgerichts',  # Federal Criminal Court
    'Bundespatentgericht', 'Bundespatentgerichts',  # Federal Patent Court
]

def should_translate(text):
    """
    Check if the text should be translated
    """
    if pd.isna(text) or text == '':
        return False
    
    # Check if text is just a proper noun or court name
    text = text.strip()
    if text in DO_NOT_TRANSLATE:
        return False
    
    # If text is very short (1-2 words), it might be a proper noun
    if len(text.split()) <= 2 and text[0].isupper():
        return False
        
    return True

def split_text_into_chunks(text, max_chunk_size=4000):
    """
    Split text into chunks of up to max_chunk_size, cutting at spaces to avoid splitting words.
    """
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        # Find the next chunk
        if n - i <= max_chunk_size:
            chunks.append(text[i:])
            break
        # Look for the last space within the chunk
        end = i + max_chunk_size
        space_pos = text.rfind(' ', i, end)
        if space_pos == -1 or space_pos <= i:
            # No space found, just cut at max_chunk_size
            chunks.append(text[i:end])
            i = end
        else:
            chunks.append(text[i:space_pos])
            i = space_pos + 1  # Skip the space
    return chunks

def translate_text(text, source_lang, target_lang='en'):
    """
    Translate text from source language to target language
    """
    if pd.isna(text) or text == '':
        return ''
    try:
        # Use improved chunking to avoid splitting words
        chunks = split_text_into_chunks(text, 4000)
        translated_chunks = []
        for chunk in chunks:
            time.sleep(0.5)
            translated = GoogleTranslator(source=source_lang, target=target_lang).translate(chunk)
            translated_chunks.append(translated)
            print(f"\nTranslation example:")
            print(f"Original ({source_lang}): {chunk[:100]}...")
            print(f"Translated (en): {translated[:100]}...")
        return ' '.join(translated_chunks)
    except Exception as e:
        print(f"Translation error: {str(e)}")
        print("Stopping script due to translation error...")
        sys.exit(1)

def process_batch(rows, output_file, is_first_batch=False):
    """
    Process a batch of rows and append to CSV file
    """
    try:
        df = pd.DataFrame(rows)
        
        # Ensure all translation columns exist
        translation_columns = ['full_text_en', 'law_area_en', 'law_sub_area_en']
        for col in translation_columns:
            if col not in df.columns:
                df[col] = ''
        
        # Verify translations
        for col in translation_columns:
            non_empty = df[col].notna().sum()
            print(f"\nVerification - {col}:")
            print(f"Total rows: {len(df)}")
            print(f"Non-empty translations: {non_empty}")
            print(f"Empty translations: {len(df) - non_empty}")
            if non_empty > 0:
                print("Sample translation:")
                sample = df[df[col].notna()].iloc[0][col]
                print(f"{sample[:200]}...")
        
        # Print column names to verify translations are present
        print("\nColumns in DataFrame:", df.columns.tolist())
        
        # Always append to the file, never overwrite
        df.to_csv(output_file, mode='a', 
                  header=not os.path.exists(output_file),  # Only write header if file doesn't exist
                  index=False, encoding='utf-8', quoting=1)
        return []
    except Exception as e:
        print(f"Error in process_batch: {str(e)}")
        print("Stopping script due to batch processing error...")
        sys.exit(1)

# Load the streaming dataset
print("Loading dataset...")
try:
    ds = load_dataset("rcds/swiss_rulings", split="train", streaming=True)
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    sys.exit(1)

# Initialize variables
output_file = 'swiss_rulings_with_translations.csv'  # Regular CSV file
batch_size = 20  # Process 100 rows at a time
current_batch = []
is_first_batch = True
total_processed = 0
total_translated = 0

# Load existing decision_ids if output file exists
existing_decision_ids = set()
if os.path.exists(output_file):
    print("Loading existing decision IDs from output file...")
    existing_df = pd.read_csv(output_file)
    existing_decision_ids = set(existing_df['decision_id'].values)
    print(f"Found {len(existing_decision_ids)} existing decision IDs")

# Process the dataset row by row
print("Processing and translating rows...")

try:
    for row in tqdm(ds):
        # Skip if decision_id already exists
        if row['decision_id'] in existing_decision_ids:
            print(f"\nSkipping existing decision_id: {row['decision_id']}")
            continue
            
        total_processed += 1
        # Create a copy of the row
        processed_row = dict(row)
        
        # Initialize translation columns
        for field in ['full_text', 'law_area', 'law_sub_area']:
            processed_row[f'{field}_en'] = ''
        
        # Translate text fields if they exist
        text_fields = ['full_text', 'law_area', 'law_sub_area']
        for field in text_fields:
            if field in row and row[field]:  # Only translate if field exists and is not empty
                print(f"\nTranslating {field} from {row['language']} to English...")
                # Add English translation
                translated = translate_text(row[field], row['language'])
                processed_row[f'{field}_en'] = translated
                if translated and translated != row[field]:  # If translation was successful
                    total_translated += 1
                    print(f"Successfully translated {field}")
                    print(f"Original: {row[field][:100]}...")
                    print(f"Translated: {translated[:100]}...")
        
        # Add to current batch
        current_batch.append(processed_row)
        
        # If batch is full, save it
        if len(current_batch) >= batch_size:
            current_batch = process_batch(current_batch, output_file, is_first_batch)
            is_first_batch = False
            print(f"\nSaved batch to {output_file}")
            print(f"Progress: {total_processed} rows processed, {total_translated} translations completed")
    
    # Save any remaining rows
    if current_batch:
        current_batch = process_batch(current_batch, output_file, is_first_batch)
        print(f"\nSaved final batch to {output_file}")
        print(f"Final stats: {total_processed} rows processed, {total_translated} translations completed")

except KeyboardInterrupt:
    print("\nProcess interrupted by user. Saving current batch...")
    if current_batch:
        current_batch = process_batch(current_batch, output_file, is_first_batch)
        print(f"Saved current batch to {output_file}")
        print(f"Progress at interruption: {total_processed} rows processed, {total_translated} translations completed")
    print("You can resume processing later by running the script again.")
    sys.exit(0)
except Exception as e:
    print(f"\nError occurred: {str(e)}")
    if current_batch:
        current_batch = process_batch(current_batch, output_file, is_first_batch)
        print(f"Saved current batch to {output_file}")
        print(f"Progress at error: {total_processed} rows processed, {total_translated} translations completed")
    print("Stopping script due to error...")
    sys.exit(1)

print("Done!")

"""
Data Fields:
decision_id (string)
facts (string)
considerations (string)
origin_facts (string)
origin_considerations (string)
law_area (string)
language (string)
year (int32)
court (string)
chamber (string)
canton (string)
region (string)
"""

