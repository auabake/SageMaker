from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "SamLowe/roberta-base-go_emotions"  # Replace with your model name

# Download and save the model
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.save_pretrained('./CustomModel1/model')

# Download and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained('./CustomModel1/model')
print("Model downloaded and saved to '.CustomModel1/model'")
