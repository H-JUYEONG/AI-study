# Hugging Face의 모델 사용
# STEP 1 : import modules
from transformers import pipeline

# STEP 2 : create inference object
classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")

# STEP 3 : prepare data
text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

# STEP 4 : inference
result = classifier(text)

# STEP 5 : post processing
classifier(text)
