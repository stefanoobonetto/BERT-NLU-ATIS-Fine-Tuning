from transformers import BertTokenizer

# Step 2: Load the BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Step 3: Tokenize the Phrase
phrase = "I'd travel from pennsilvania to johannesburg"
tokens = tokenizer.tokenize(phrase)

# Step 4: Convert Tokens to IDs
ids = tokenizer.convert_tokens_to_ids(tokens)

print("Original phrase:", phrase)
print("Tokens:", tokens)
print("IDs:", ids)