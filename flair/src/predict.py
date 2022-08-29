from os import path

from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
from transformers.models.auto.tokenization_auto import AutoTokenizer


def main():
    models_path = path.join(path.dirname(__file__), "../models")
    model_path = path.join(models_path, "test")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    text = """
        Augusta Ada King, condesa de Lovelace (Londres, 10 de diciembre de 1815-íd., 27 de noviembre de 1852), registrada al nacer como Augusta Ada Byron y conocida habitualmente como Ada Lovelace, fue una matemática y escritora británica, célebre sobre todo por su trabajo acerca de la computadora mecánica de uso general de Charles Babbage, la denominada máquina analítica. Fue la primera en reconocer que la máquina tenía aplicaciones más allá del cálculo puro y en haber publicado lo que se reconoce hoy como el primer algoritmo destinado a ser procesado por una máquina, por lo que se la considera como la primera programadora de ordenadores
    """

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs)

    # Get the result.
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(prediction)


if __name__ == "__main__":
    main()
