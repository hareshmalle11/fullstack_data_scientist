# Cat vs Dog Classifier

This project uses a trained TensorFlow model and a Streamlit front end to classify uploaded images as either a cat or a dog.

## Files

- `app.py` - Streamlit app
- `cat_dog_classifier.keras` - trained model file expected in the same folder as `app.py`
- `requirements.txt` - Python dependencies
- `cnn_pet.ipynb` - model development notebook

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

Make sure the trained model file is saved as `cat_dog_classifier.keras` in the project root.

## Run the app

Start the Streamlit app:

```bash
streamlit run app.py
```

## Notes

- The model expects images resized to `128 x 128`.
- The binary output is mapped as `0 = Cat` and `1 = Dog`.
- If the model file is missing, the app will show an error message.
