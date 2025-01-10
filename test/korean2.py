# STEP 1 : import modules
import easyocr

# STEP 2 : create inference object
reader = easyocr.Reader(
    ["ko", "en"]
)  # this needs to run only once to load the model into memory

# STEP 3 :load data
data = "3.jpg"

# STEP 4 : inference
result = reader.readtext(data, detail=0)
print(result)

# STEP 5: post processing
