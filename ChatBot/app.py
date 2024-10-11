from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import uvicorn
from prompts import prompt_template_for_question
import re

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model_name="llama-3.1-70b-versatile", api_key=api_key)

df = pd.read_csv('data/grocery_data.csv')
filtered_df = df[['description', 'categoryName', 'categoryID', 'stock', 'price',
       'nutritions','img','name']]
#df = SmartDataframe(df, config={"llm": llm})
def check_if_data_needed(user_input: str) -> bool:
    prompt = """The user has asked: "{user_input}".
    Based on this question, determine if the query needs to use the Grocery dataset
    or if it's just a general question. Respond with 'yes' if it needs data, 
    otherwise respond with 'no'.
    If query is asking about grocery products than respond with 'yes' esle 'no'.
    Output should be either 'yes' or 'no'. """
    
    prompt = PromptTemplate(template=prompt, input_variables=["user_input"])
    chain = prompt | llm
    
    decision = chain.invoke({"user_input": user_input})
    decision = decision.content.strip().lower()
    return decision == 'yes'


def get_data_from_csv(user_query: str) -> str:
    prompt = PromptTemplate(template=prompt_template_for_question, input_variables=["user_query"])
    chain = prompt | llm

    output = chain.invoke({"user_query": user_query})
    pattern = r"Python Code: ```(.*?)```"
    print(output.content, "--------output content---------------")
    matches = re.findall(pattern, output.content, re.DOTALL)
    print(matches,"------------------matches code----------------")
    
    if matches:
        
        python_code = matches[0]
        try:
            # Execute the Python code using eval() to retrieve data from 'df'
            result = eval(python_code)
            json_data = result.to_json(orient='records')
            return json_data
        except Exception as e:
            return f""
    else:
        return ""

def respond_to_user(user_input: str) -> str:
    needs_data = check_if_data_needed(user_input)
    
    if needs_data:
        retrieved_data = get_data_from_csv(user_input)
        prompt = PromptTemplate(
            template="""You are a helpful Grocery Store Assistant. Respond to the user's query: '{user_input}' and retrieved data: '{retrieved_data}'.
            Anwer should be based on the retrieved data, Don't add extra from your side. Answer should be good as a Assistant. 
            You should offer and give details about the products from retrived data.
            
            """,
            input_variables=["user_input","retrieved_data"]
        )
        chain = prompt | llm
        output = chain.invoke({"user_input": user_input,"retrieved_data": retrieved_data})
        response = output.content
        data = {"ai_response": response, "data": retrieved_data}
        print("-----------------------final Output:  ",data)
        return data
    else:
        print("without retrieved data output ------")
        prompt = PromptTemplate(
            template="You are a helpful Grocery Store Assistant. Respond to the user's query: '{user_input}'. Don't add prducts details from your side.",
            input_variables=["user_input"]
        )
        chain = prompt | llm
        output = chain.invoke({"user_input": user_input})
        response = output.content
        data = {"ai_response": response, "data": ""}
        return data

class QueryRequest(BaseModel):
    query: str

app = FastAPI()

@app.post("/query/")
async def handle_query(request: QueryRequest):
    try:
        user_input = request.query
        response = respond_to_user(user_input)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
