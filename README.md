# Poetry Generation Model

This project involves creating a poetry generation model using deep learning techniques. The model leverages Long Short-Term Memory (LSTM) networks to generate creative and coherent poems based on a given seed text. The approach uses TensorFlow and Keras to build and train the model on a dataset of poems, aiming to capture poetic structures and stylistic nuances.

## Features

- **Text Generation**: Generate new poetry based on a seed text.
- **Customizable Training**: Ability to train the model on different datasets.
- **Pre-trained Model**: Includes a pre-trained model for immediate use.
- **Metrics**: Includes custom perplexity metric to evaluate model performance.

## Requirements

- Python 3.6 or higher
- TensorFlow 2.x
- Keras
- Pandas
- NumPy
- Matplotlib


## How It Works

1. **Data Preparation**: The model is trained on a dataset of poems, with preprocessing steps to clean and format the text data.
2. **Model Architecture**: The model uses an embedding layer to convert words into dense vectors, followed by two LSTM layers to capture temporal dependencies in the text.
3. **Training**: The model is trained using a sequence of poems, learning to predict the next word in the sequence.
4. **Generation**: The trained model can generate new poetry by predicting the next word given a seed text.

## Example

Here is an example of how to generate a poem using the pre-trained model:

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the tokenizer and model
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(['example text'])  # Replace with actual text for fitting
model = load_model('poet_model_v3.h5')

def generate_poem(seed_text, num_of_words):
    poem = seed_text

    for i in range(num_of_words):
        sequences = tokenizer.texts_to_sequences([seed_text])
        padded_sequences = pad_sequences(sequences, maxlen=40, truncating='pre')

        prediction = model.predict(padded_sequences)
        predicted_word_index = np.argmax(prediction, axis=-1)[0]
        predicted_word = tokenizer.index_word.get(predicted_word_index, '')

        poem += ' ' + predicted_word
        seed_text = ' '.join(seed_text.split()[1:] + [predicted_word])

    return poem


seed_text = "The night is dark and full of stars"
generated_poem = generate_poem(seed_text, 20)
print(generated_poem)
```
## Known Issues

- **Model Performance**: The quality of generated poems may vary based on the training data and model hyperparameters. The current model may produce less coherent or stylistically inconsistent results. Further tuning and training with a more diverse dataset could improve performance.

- **Data Handling**: Ensure that the input data is clean and formatted correctly to avoid errors during training and generation. Inconsistent or improperly formatted data can lead to issues in text generation.

- **Training Time**: Training the model can be time-consuming, especially with large datasets and complex architectures. Consider using a GPU to speed up the training process if available.

- **Dependency Management**: Make sure all required packages and dependencies are correctly installed. Version mismatches between TensorFlow, Keras, and other libraries may cause compatibility issues.

## Future Improvements

- **Data Augmentation**: Incorporate more diverse and extensive datasets to improve the model’s ability to generate varied and high-quality poetry. This could involve adding poems from different genres and authors.

- **Hyperparameter Tuning**: Experiment with different hyperparameters such as learning rates, batch sizes, and the number of LSTM units to optimize model performance. Consider using techniques like grid search or random search for hyperparameter optimization.

- **Model Extensions**: Explore advanced architectures such as Transformer models (e.g., GPT-2, GPT-3) which have shown improved performance in text generation tasks. This could lead to better quality and more coherent generated text.

- **User Interface**: Develop a web or desktop application to make the poetry generation model more accessible to users. This could include features for customizing seed text and generating poems interactively.

- **Evaluation Metrics**: Implement additional evaluation metrics such as BLEU score or ROUGE score to quantitatively assess the quality of generated poems. These metrics can help in comparing different models and improvements.

## Contact

For any questions, feedback, or contributions, please reach out to bocrar@gmail.com


## Acknowledgements

- **TensorFlow** and **Keras** for providing the deep learning frameworks.
- The contributors of the datasets used for training and evaluation https://www.kaggle.com/datasets/ramjasmaurya/poem-classification-nlp/data.

