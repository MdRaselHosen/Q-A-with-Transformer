from django.shortcuts import render
import os
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

Model_path = "model"
tokenizer = AutoTokenizer.from_pretrained(Model_path)
model = AutoModelForQuestionAnswering.from_pretrained(Model_path)
model.eval()
torch.set_grad_enabled(False)

def predict_answer(context,question):
    inputs = tokenizer(question,context,return_tensors="pt",truncation=True,max_length=512)
    outputs = model(**inputs)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits) + 1

    answer = tokenizer.decode(
        inputs["input_ids"][0][start_index:end_index],
        skip_special_tokens=True
    )

    return answer
def home(request):
    answer = ""
    if request.method == "POST":
        context = request.POST.get("context")
        question = request.POST.get("question")

        if context and question:
            answer = predict_answer(context, question)

    return render(request, "home.html", {"answer":answer})