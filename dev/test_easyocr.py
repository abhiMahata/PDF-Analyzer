import fitz  
import easyocr
import time




try:
    print("Loading EasyOCR...")
    start = time.time()
    reader = easyocr.Reader(['en'], gpu=False) # Fallback to CPU if no GPU available
    print(f"Loaded in {time.time() - start:.2f} seconds.")
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
