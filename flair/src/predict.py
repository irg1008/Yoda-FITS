from os import path

from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
from transformers.models.auto.tokenization_auto import AutoTokenizer


def main():
    models_path = path.join(path.dirname(__file__), "../models")
    model_path = path.join(models_path, "test")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    text = """
    Vladimir Ilyich Ulyanov[b] (22 April [O.S. 10 April] 1870 – 21 January 1924), better known as Vladimir Lenin,[c] was a Russian revolutionary, politician, and political theorist. He served as the first and founding head of government of Soviet Russia from 1917 to 1924 and of the Soviet Union from 1922 to 1924. Under his administration, Russia, and later the Soviet Union, became a one-party socialist state governed by the Communist Party. Ideologically a Marxist, his developments to the ideology are called Leninism."""
    
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs)

    # Get the result.
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(prediction)


if __name__ == "__main__":
    main()
