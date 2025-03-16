import joblib
import pytesseract
import pandas as pd
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
# Load the saved model
knn_model = joblib.load('ocr_knn_model.pkl')

def process_image(image):

    # Perform OCR
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    final_df = pd.DataFrame(ocr_data)


    # Compute Bounding Box Details
    final_df['bottom'] = final_df['top'] + final_df['height']
    final_df['right'] = final_df['left'] + final_df['width']

    # Select feature columns
    feature_columns = ["left", "top", "width", "height", "bottom", "right"]
    X_test = final_df[feature_columns]

    # Predict the labels for the text
    final_df["predicted_label"] = knn_model.predict(X_test)

    # Gather all records labeled as 1 and convert the list into a string
    address = ' '.join(final_df.loc[final_df['predicted_label'] == 1]['text'].to_list())

    return address
