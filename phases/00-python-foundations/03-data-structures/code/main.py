import re
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "The cat sat on the mat.",
    "The dog sat on the log.",
    "Cats and dogs are friends, and the cat likes the dog.",
    "A model learns patterns from data.",
]


def tokenize(text):
    return re.findall(r"\b\w\w+\b", text.lower())


print("Step 1: lists hold ordered tokens")
tokenized_documents = [tokenize(document) for document in documents]
for index, tokens in enumerate(tokenized_documents):
    print(f"  doc {index}: {tokens}")
first_tokens = tokenized_documents[0]
print(f"  first token {first_tokens[0]!r}, last token {first_tokens[-1]!r}, slice [1:3] {first_tokens[1:3]}")

print()
print("Step 2: sets give unique words and fast membership")
vocabulary_set = set()
for tokens in tokenized_documents:
    vocabulary_set.update(tokens)
vocabulary = sorted(vocabulary_set)
print(f"  vocabulary size {len(vocabulary)}")
print(f"  'cat' in vocabulary -> {'cat' in vocabulary_set}, 'bird' in vocabulary -> {'bird' in vocabulary_set}")
words_doc0 = set(tokenized_documents[0])
words_doc1 = set(tokenized_documents[1])
print(f"  shared by doc 0 and doc 1   {sorted(words_doc0 & words_doc1)}")
print(f"  in doc 0 but not in doc 1   {sorted(words_doc0 - words_doc1)}")


def jaccard_similarity(first_words, second_words):
    union = first_words | second_words
    if not union:
        return 0.0
    return len(first_words & second_words) / len(union)


print(f"  Jaccard(doc 0, doc 1) = {jaccard_similarity(words_doc0, words_doc1):.3f}")
print(f"  Jaccard(doc 0, doc 3) = {jaccard_similarity(words_doc0, set(tokenized_documents[3])):.3f}")

print()
print("Step 3: dicts map words to indices and counts")
word_to_index = {word: index for index, word in enumerate(vocabulary)}
print(f"  word_to_index['cat'] = {word_to_index['cat']}, word_to_index['the'] = {word_to_index['the']}")

word_frequencies = {}
for tokens in tokenized_documents:
    for token in tokens:
        word_frequencies[token] = word_frequencies.get(token, 0) + 1
top_words = sorted(word_frequencies.items(), key=lambda pair: (-pair[1], pair[0]))[:5]
print(f"  top 5 words {top_words}")

print()
print("Step 4: a bag-of-words matrix as a list of lists")


def bag_of_words(tokenized, index_of_word):
    matrix = []
    for tokens in tokenized:
        row = [0] * len(index_of_word)
        for token in tokens:
            if token in index_of_word:
                row[index_of_word[token]] += 1
        matrix.append(row)
    return matrix


count_matrix = bag_of_words(tokenized_documents, word_to_index)
header = " ".join(f"{word[:4]:>4}" for word in vocabulary[:10])
print(f"         {header} ...")
for index, row in enumerate(count_matrix):
    print(f"  doc {index}: " + " ".join(f"{count:>4}" for count in row[:10]) + " ...")

print()
print("Step 5: tuples as immutable records")
document_records = [(index, len(tokens), tokens[0]) for index, tokens in enumerate(tokenized_documents)]
for doc_id, length, first_word in document_records:
    print(f"  doc {doc_id}: {length} tokens, starts with {first_word!r}")
longest = max(document_records, key=lambda record: record[1])
print(f"  longest document: {longest}")

print()
print("Use It: collections.Counter and sklearn CountVectorizer")
counter = Counter(token for tokens in tokenized_documents for token in tokens)
print(f"  Counter.most_common(5) {counter.most_common(5)}")
print(f"  Counter matches our dict: {dict(counter) == word_frequencies}")

vectorizer = CountVectorizer()
sparse_counts = vectorizer.fit_transform(documents)
sklearn_vocabulary = list(vectorizer.get_feature_names_out())
print(f"  same vocabulary: {sklearn_vocabulary == vocabulary}")
print(f"  same count matrix: {sparse_counts.toarray().tolist() == count_matrix}")
print(f"  sklearn stores it sparsely: {sparse_counts.nnz} non-zeros out of {sparse_counts.shape[0] * sparse_counts.shape[1]} cells")
