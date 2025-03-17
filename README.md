# Computer-Vision-OCR
-final project for Rugters AI bootcamp

In this git we address the issue of W2 document scanning that is usable on a local consumer device. The desired features to be extracted from the W2 are the address, wages, first and last name, and year to be used for personal record keeping. Using a query based Donut transformer, other features of the document can be extracted but with limited accuracy. To address the issues with extracting the employee address, a separate ocr model is engaged and employed when it detects "employee" and "address" in the query. Because Donut is not a massive transformer model it is recommended to use the vocabulary that is as close to what is present on the document as possible. Technically the code is device agnostic and a cpu can be used, however a computer with a GPU having at least 19 GB of free vram and reasonably fast clock speeds was employed in testing (Radeon 7900xtx), using this GPU the processing of 85 documents via the document parser took ~4 minutes, cpu will probably be much slower. 

## Fast Start

$ python gradio_combo_adventure.py, type in "y" to run the GUI then click on the resultant address from which to run the gradio app from (it is set to be local). 
Example queries:
"What is the Employee's address?"
"What is the wage?"
"What is the Employee's first name and last name?"

Provide a query and upload the image of the document when prompted. 

## Slower Start
$ python gradio_combo_adventure.py, type in "n" to run the document name, wage, address, and year extractor. You must provide a path to the document images that the program can access. The end result is a csv file named "names_wages_add_years_w2.csv"

### Important considerations
The model is far from perfect, expect misspellings, dropped words from addresses, and other odd errors including the replacing of the number '1' with the letter 'i'. Using a manual inspection scoring system where error points are added in accordance with type of mistake (+0.25 for minor mistake, +0.5 for a dropped word, and +2 for a completely wrong answer, i.e. lower is better and some minor mistakes are 'forgivable') and then averaging the error number per query the combined model can achieve an error performance of 0.341 across 10 test W2 documents when the lone Donut model achieved a 0.685. To see the testing performance consult the csvs titled "none_cornercase_resolver" and "yes_cornercase_resolver" in which queries with model answers and expected answers are recorded during a ten document test run. In the "none" file the Donut model is run without the employee address ocr model. In the "yes" file the Donut model is run with the employee address ocr model. 

### To-Do
Fine-tune the Donut model to the W2 specific format, include certain transformations of the image (such as rotation, scaling, etc.) to improve generalized performance especially with scanned W2s.
Utilize a different W2 dataset, this dataset while very detailed, has an issue where the zip codes are very inconsistent which is why only the address without the zip-code is used. It would also be good to include at least a few W2's with handwriting. Alternatively create a W2 dataset generator to create both the W2 and the corresponding json of correct answers from queries.
Should the fine-tuning of the Donut model fail to improve the extraction of data, replace the Donut transformer with another model or in addition to the address extraction model, train another one that is employed when certain queries are called. 

#### References:
W2 dataset from: https://www.kaggle.com/datasets/mcvishnu1/fake-w2-us-tax-form-dataset
Donut transformer model from: https://huggingface.co/docs/transformers/main/en/model_doc/donut, the specific model used is: "naver-clova-ix/donut-base-finetuned-docvqa"

#### Requirements to run base gradio_combo_adventure:
gradio==5.21.0

joblib==1.4.2

pandas==2.2.3

Pillow==11.1.0

pytesseract==0.3.13

torch==2.6.0

tqdm==4.66.5

transformers==4.49.0

along with any additional dependencies required by the above libraries.

