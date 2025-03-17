
import re
from transformers import DonutProcessor
from transformers import VisionEncoderDecoderModel
import torch
import os
from accelerate.utils.memory import clear_device_cache
import py_knn_ocr_modelrunner as pm
import gradio as gr
import pandas as pd
import json
from PIL import Image as PImage
import glob
from tqdm import tqdm
os.environ['PYTORCH_HIP_ALLOC_CONF']='expandable_segments:True'

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print('Using device:', device)
print()

#After searching a bit, Donut seems to be the best OCR replacement. This version the finetuned-docvqa is ideal for more generalized 'find thing in pdf' usage. 
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

model.to(device)

#This method is to load jpg images when given a path, the test file path is set as default.
def load_img(path="Input/W2_Single_Clean_jpg/"):
    files = glob.glob(f"{path}/*.jpg")
    loaded_images = []
    for f in files:
        img = PImage.open(f)
        loaded_images.append(img)
    return loaded_images


#This method is to load the testing/training image with json sets.
def load_img_with_json(path="Input/W2_Single_Clean_jpg/", start=0, stop=46):
    json_path = path + "jsons_data/"
    img_file_base = path + "W2_XL_input_clean_"
    json_file_base = json_path + "W2_"
    image_file_names = []
    #The start and stop are manufally input to parse through the test files that are known to have corresponding jsons.
    jsons = []
    images = []
    for i in range(start, stop):
        number_to_load = 1000 + i
        json_to_load = json_file_base + str(number_to_load) + ".json"
        img_to_load = img_file_base + str(number_to_load) + ".jpg"
        try:
            with open(json_to_load, 'r') as file:
                json_loaded = json.load(file)
                jsons.append(json_loaded)
        except:
            print(json_to_load)
        img = PImage.open(img_to_load)
        
        image_file_names.append(img_to_load)
        images.append(img)

    return images, jsons, image_file_names
        


def question_result(question:str, image, corner_employee=True):
    """
    This function is used to parse information from a W2. To deal with the wonky results from the employee's address box we utilize a model trained on retrieving the correct data from the bounding box
    question: a query in string format
    image: an image in jpg format
    corner_employee: a bool which determines whether the corner case "employee address" model is used. To test Donut's lone performance set this value to False. 
    """
    results = []
    result = ''
    if ("address" in question.lower()) and (("employee's" in question.lower()) or ("employees" in question.lower()) or ("employee" in question.lower())) and corner_employee:
        result = pm.process_image(image)
    else:
        task_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"

        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

        pixel_values = processor(image, return_tensors="pt").pixel_values

        outputs = model.generate(

            pixel_values.to(device),

            decoder_input_ids=decoder_input_ids.to(device),

            max_length=model.decoder.config.max_position_embeddings,
    
            pad_token_id=processor.tokenizer.pad_token_id,

            eos_token_id=processor.tokenizer.eos_token_id,

            use_cache=True,

            bad_words_ids=[[processor.tokenizer.unk_token_id]],

            return_dict_in_generate=True

        )

        sequence = processor.batch_decode(outputs.sequences)[0]

        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")

        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
        result = processor.token2json(sequence)['answer']
        results.append(result)
        print(f"{question} : {result}")
        
    return result

def sanity_test():
    images, jsons, image_names = load_img_with_json(path="Input/W2_Test_Employee_Address/", start=88, stop = 98)
    #Now we will get an idea how how good the estimates are by loading from the test data 
    queries = ["What is the Employee's Address?", "What is the Employee's first name and last name?", "What are the wages, tips and compensation?", "What is the first value for the state income tax in box 17?", "What is the year next to the Wage and Tax Statement?"]
    json_collect = ["Employee's address", "Employee's first name and last name", "Wages", "box_17_1"]
    questions = []
    answers = []
    expecteds = []
    for idx, img in enumerate(images):
        for idxq, query in enumerate(queries):
            result = question_result(query, img, False)
            #All of the W2 forms are of the year 2010
            expected = "2010"
            if idxq < len(json_collect):
                expected = jsons[idx][json_collect[idxq]]
            questions.append(query)
            answers.append(result)
            expecteds.append(expected)
    no_corners = pd.DataFrame({"queries":questions, "model_answers":answers, "actual_answers":expecteds})
    no_corners.to_csv("none_cornercase_resolver.csv")

    questions = []
    answers = []
    expecteds = []
    for idx, img in enumerate(images):
        for idxq, query in enumerate(queries):
            result = question_result(query, img)
            #All of the W2 forms are of the year 2010
            expected = "2010"
            if idxq < len(json_collect):
                expected = jsons[idx][json_collect[idxq]]
            questions.append(query)
            answers.append(result)
            expecteds.append(expected)
    yes_corners = pd.DataFrame({"queries":questions, "model_answers":answers, "actual_answers":expecteds})
    yes_corners.to_csv("yes_cornercase_resolver.csv")

def name_add_wages(img):
    queries = ["What is the Employee's Address?", "What is the Employee's first name and last name?", "What are the wages, tips and compensation?", "What is the year next to the Wage and Tax Statement?"]
    label = ['address', 'name', 'wages', 'year']
    results = {}
    for idx, q in enumerate(queries):
        answer = question_result(q, img)
        results[label[idx]] = answer
    return results    
    
def multi_doc_parser(path="Input/W2_Single_Clean_jpg/"):
    images = load_img(path)
    names = []
    addresses = []
    wages = []
    years = []
    for img in tqdm(images):
        res = name_add_wages(img)
        names.append(res['name'])
        addresses.append(res['address'])
        wages.append(res['wages'])
        years.append(res['year'])
    results = pd.DataFrame({"names":names, "wages":wages, "years":years, "addresses":addresses})
    results.to_csv("names_wages_add_years_w2.csv")

def run_gradio():
     gr.Interface(
        fn=question_result,
        inputs=[gr.Textbox(type='text'),gr.Image(type="numpy")],
        outputs="text",
        title="What's in my document?",
        description="Upload an image of a document and a question to extract a desired feature of the document"
    ).launch()



if __name__ == "__main__":
    decision = input("Run GUI (Y/N?): ")
    acceptable_answers = ['y', 'yes', 'n', 'no']
    while (not(decision.lower() in acceptable_answers)):
        decision = input("Please input 'Y' or 'N'")

    gradio_mode = False
    if decision.lower() in acceptable_answers[:1]:
        gradio_mode = True
    
    if gradio_mode:
        run_gradio()
    else:
        path = input("What is the path to your document image files in jpg?: \n")
        multi_doc_parser(path)
