# بِسْمِ ٱللَّٰهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ و بهِ نَستَعين

import spacy
import gensim.downloader as api

model = api.load("glove-wiki-gigaword-100")
class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
for cls in class_names:
    if cls in model:
        print(model[cls])
    else:
        print(f"{cls} does not exist")


# Load a pre-trained language model:
nlp = spacy.load("en_core_web_trf")  # More accurate (needs a GPU)
nlp = spacy.load("en_core_web_sm")  # More efficient

# Define the text you want to convert:
text = "This is an example text."

word = "example"
if word in nlp.vocab:
    print(f"'{word}' is in the vocabulary")
else:
    print(f"'{word}' is not in the vocabulary")

# Process the text with the language model:
doc = nlp(text)

# Extract the word vectors for each word in the text:
word_vectors = [word.vector for word in doc]
print(word_vectors)

# To extract the word vectors for each word in a list:
class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
vectors = [nlp(name).vector for name in class_names]
print(vectors)


# class_names = data.classes
# class_to_vec = {}
# matrix = np.empty((len(class_names), 300))
# for i, class_name in enumerate(class_names):
#     vector = np.mean([model[cls] for cls in class_name.lower().split(' ')], 0)
#     class_to_vec[class_name] = vector
#     matrix[i, :] = vector


# df = pd.DataFrame(matrix, index=class_names)
# print(df)
# df.to_csv("data_vectors.csv")
