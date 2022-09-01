from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from model import get_model_and_tokenizer


def main():
    model, tokenizer = get_model_and_tokenizer()

    text = """
        HUAWEI Watch FIT - Smartwatch con Cuerpo de Metal, Pantalla AMOLED de 1.64”, hasta 10 días de batería, 96 Modos de Entrenamiento, GPS Incorporado, 5ATM, Color Rojo
    """

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs)

    # Get the result.
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(prediction)


if __name__ == "__main__":
    main()
