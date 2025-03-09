# Computer-Vision-OCR

The input files used in this workflow are not added as for security reasons of the documents. 


## End to End Steps
**Perform Preprocessing Steps**
- Convert documents (e.g., PDFs) into images.
- Extract text and bounding boxes using an OCR engine like Tesseract.
- Normalize bounding boxes to fit the 0-1000 range for LayoutLMv3.
- Clean and correct OCR errors (e.g., fuzzy matching for misspellings).

**Fine-Tune LayoutLMv3 on the Custom Dataset**
- Prepare the dataset in Hugging Face's datasets format (words, bounding boxes, labels).
- Train LayoutLMv3 using labeled data (assign QUESTION, ANSWER, HEADER, etc.).
- Evaluate the model’s performance on validation data.

**Process and Use the Trained Model**
- Load the fine-tuned LayoutLMv3 model.
- Pass new document images and extracted text through the model.
- Retrieve structured output with correctly labeled entities.


## Tokenization
Encoding = processor.tokenizer(words, return_tensors=”pt”)
Returns a dictionary with keys such a ‘bbox’, ‘position_ids’

Tokenization process in LayoutLMv3 involves: 

1. Since LayoutLMv3 uses Byte-Pair Encoding(BPE) from RoBERTa, words are split into smaller subwords
	
	Example: "DOCUMENT" → ["DOC", "UMENT"] (how words are split)

2. Groups tokens into sequences based on their bounding box positions (spatial relationship)
    - If words are on the same line, they are grouped together.
    - If a word is far apart, it starts a new sequence.

	To visualize token positions : `print(encoding[“bbox”])`


3. Assigns position IDs to each sequence based on the document structure (position embeddings) 

	To visualize position embeddings: `print(encoding[“position_ids”])`
