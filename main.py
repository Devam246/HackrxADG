#uvicorn main:app --reload ; this is very good line, it reload/starts
#the server again


# main.py


# 1. Import necessary libraries
#    - FastAPI: The main framework
#    - Pydantic's BaseModel: For creating data models (schemas)
#    - List, Optional: For type hinting
#    - HTTPException, status: For handling errors
#    - Header, Depends: For dependency injection (used for the auth token)
from fastapi import FastAPI, HTTPException, status, Header, Depends
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from rag_pipeline import extract_chunks_from_any_file, handle_queries
import time
import os
from dotenv import load_dotenv
load_dotenv()

EXPECTED_BEARER_TOKEN = os.getenv("HACKATHON_BEARER_TOKEN")

# --- Data Models (using Pydantic) ---
# These models define the exact structure of your request and response JSON.
# FastAPI uses them to validate incoming data and serialize outgoing data.


"""
Basically we have defined a class here
Whats class, harry ki video dekhi hogi to pata chal gya hoga, anyways will explain here qckly
Analogy, its 99% same as that of structure in cpp


"""
class HackathonRequest(BaseModel):
   """This model defines the structure of the JSON you expect to receive."""
   documents: HttpUrl  # Pydantic validates this is a valid URL
   questions: List[str] # list is a DS in python


class HackathonResponse(BaseModel):
   """This model defines the structure of the JSON you will send back."""
   answers: List[str]


# --- Authorization ---
# This is a dependency that will check the Authorization header.


# The required token for the hackathon. Store it securely.
# For now, we'll hardcode it here. In a real app, you'd use environment variables.



async def verify_token(authorization: Optional[str] = Header(None)):
   """
   This function is a "dependency". FastAPI will run it before your main logic.
   It checks if the Authorization header is present and correct.
   """
   if authorization is None:
       raise HTTPException(
           status_code=status.HTTP_401_UNAUTHORIZED,
           detail="Authorization header is missing",
       )
  
   # The header value should be "Bearer <token>"
   parts = authorization.split()
   if len(parts) != 2 or parts[0].lower() != "bearer":
       raise HTTPException(
           status_code=status.HTTP_401_UNAUTHORIZED,
           detail="Invalid authorization header format. Must be 'Bearer <token>'",
       )
  
   token = parts[1]
   if token != EXPECTED_BEARER_TOKEN:
       raise HTTPException(
           status_code=status.HTTP_403_FORBIDDEN,
           detail="Invalid token",
       )
   print("📛 Provided token:", token)
   print("🔒 Expected token:", EXPECTED_BEARER_TOKEN)

   # If we reach here, the token is valid.
   # We don't need to return anything. If an exception isn't raised, FastAPI proceeds.


# --- FastAPI App Instance ---
"""
Now, u will thik, FASTAPI class is used but where we have defined this class,
actually we have imported it earlier
Analogy : we write cout << " hell(o) " << endl;
where is the code of cout, we imported it from #include<iostream>
"""
app = FastAPI(
   title="HackRx RAG API",
   description="An API to answer questions based on a given document.",
   version="1.0.0"
)




# --- API Endpoint ---
# @ sign is decorator
# .post is the method

@app.post(
   "/hackrx/run", #its url path for this endpoint
   response_model=HackathonResponse, #we have defined the HackathonResponse, reponse must
   # look like in that format, like what we have defined earlier already
   summary="Run the RAG system", #creates short ह्यूमन readable title for the endpoint
   tags=["RAG"]
)



async def run_rag_system(
   request_data: HackathonRequest,
   _: None = Depends(verify_token)
):
   print("📥 API HIT: Received request to /hackrx/run")  # ← Add this line
   """
   Main endpoint to handle document URL + questions and return LLM answers.
   """
# 1. Chunk the document (this now returns chunks AND a doc_id)
#    The doc_id is essential for caching the next steps.


   try:
       start = time.perf_counter()
       print(f"Received document URL: {request_data.documents}")
       print(f"Total questions: {len(request_data.questions)}")

       # 1. Chunk the document (PDF, DOCX, EML)
       chunks, doc_id = extract_chunks_from_any_file(str(request_data.documents))
       #chunks = extract_chunks_from_any_file(str(request_data.documents))

       # 2. Use RAG to get answers
       answers = handle_queries(
           queries=request_data.questions,
           chunks=chunks,
           doc_id=doc_id, # Pass the unique document ID here
           top_k=3  # Adjust this based on your needs
       )
       total_time = time.perf_counter() - start
       print(f"🚀 True Total API Time: {total_time:.2f} seconds")
       return HackathonResponse(answers=answers)

#    except Exception as e:
#        print(f"❌ Error: {str(e)}")
#        raise HTTPException(status_code=500, detail="Internal server error while processing document.")
   except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while processing document.")


   
#    This endpoint performs Retrieval-Augmented Generation.


#    - It receives a document URL and a list of questions.
#    - It processes them to generate answers.
#    - **Authorization**: Requires a valid Bearer token.
   
  
   # At this point, FastAPI has already:
   # 1. Validated that the incoming JSON matches the `HackathonRequest` model.
   # 2. Verified the Authorization token using our `verify_token` dependency.


   # --- YOUR RAG LOGIC GOES HERE ---
   # This is where you will implement the core functionality of your hackathon project.
  
   # 1. Download the PDF from the `request_data.documents` URL.
   # 2. Load and split the PDF into text chunks.
   # 3. Generate embeddings for each chunk.
   # 4. Store the chunks and their embeddings in a vector database (like Pinecone).
   # 5. For each question in `request_data.questions`:
   #    a. Generate an embedding for the question.
   #    b. Query the vector DB to find the most relevant text chunks.
   #    c. Combine the question and the relevant chunks into a prompt for an LLM (like GPT-4).
   #    d. Get the answer from the LLM.
   # 6. Collect all the answers.
  
  

# To run this app:
# 1. Save the code as `main.py`.
# 2. Make sure you have the necessary libraries:
#    pip install "fastapi[all]"
# 3. Run the server from your terminal:
#    uvicorn main:app --reload



