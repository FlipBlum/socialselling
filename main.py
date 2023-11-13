import os
from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# 0. API Keys
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
proxycurl_api_key = os.getenv("PROXYCURL_API_KEY")

# 1. Tool für die Suche
def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response.text


def get_profile(profile_id):
    api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
    header_dic = {'Authorization': 'Bearer ' + proxycurl_api_key}
    params = {
        'url': f'https://www.linkedin.com/in/{profile_id}',
    }
    response = requests.get(api_endpoint,
                            params=params,
                            headers=header_dic)
    
    return response.json()

def get_company(company_id):
    api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
    header_dic = {'Authorization': 'Bearer ' + proxycurl_api_key}
    params = {
        'url': f'https://www.linkedin.com/company/{company_id}',
    }
    response = requests.get(api_endpoint,
                            params=params,
                            headers=header_dic)
    
    return response.json()
    
# 3. Tool fürs Zusammenfassen
def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output

# 5. E-Mail generierung

def generate_linkedIn_email(prompt):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    # Erstelle das Prompt, das alle Informationen enthält
    prompt = """
    Hier sind die Informationen zur E-Mail:

    **Instructions:**
    - Hier sind Informationen zu dem Unternehmen von dem aus die E-Mail losgeschickt wird:
    Wir sind die Schlander & Blum GmbH. Wir verkaufen Qualitäts, Rework und Montage Outsourcingkapazitäten an Unternehmen aller Branchen. 
    Bei Schlander & Blum können sie sich auf höchste Qualität verlassen, sei es mit Automatisierungstechnik, Sondermaschinenbau oder Menschen, wir liefern das was sie brauchen und mehr.

    **Company Summary:**
    {company_summary}

    **Person Summary:**
    {person_summary}

    **Additional Information:**
    {weitere_informationen}

    **Generiere eine E-Mail für Social Selling auf LinkedIn:**
    """

    # Kommuniziere mit dem LLM und erhalte die generierte E-Mail
    generated_email = llm.generate(prompt, max_tokens=200)

    return generated_email

# 4. Streamlit Benutzeroberfläche

st.title("LinkedIn Scraper & LLM Text Generation")

weitere_informationen = st.text_input("Weitere Informationen")
company_url = st.text_input("Unternehmens-URL")
company_linkedin_url = st.text_input("LinkedIn-URL (Unternehmen)")
linkedin_url = st.text_input("LinkedIn-URL (Person)")

if st.button("Start"):
    # Schritt 1: Suchanfragen
    company_search = search("site:linkedin.com " + company_url)
    company_linkedin_search = search("site:linkedin.com/company " + company_linkedin_url)


    def perform_scraping(): 
    # Use Proxycurl API to get profile data
        company_content = get_profile(company_url)
        company_linkedin_content = get_company(company_linkedin_url)
        person_content = get_profile(linkedin_url)
    
        return company_content, company_linkedin_content, person_content
        
    if company_content and company_linkedin_content and person_content:
    
        # Schritt 4: Anzeige der Zusammenfassungen
        st.subheader("Zusammenfassung zur Person:")
        st.write(person_content)

        st.subheader("Zusammenfassung zum Unternehmen:")
        st.write(company_content)

        # Anzeige der zusätzlichen Informationen
        st.subheader("Weitere Informationen:")
        st.write(weitere_informationen)

        st.subheader("Generierte E-Mail:")
        # Generiere die E-Mail mit dem erstellten Prompt
        generated_prompt = f"Company Summary: {company_content}\nPerson Summary: {person_content}\nAdditional Information: {weitere_informationen}\n\nGeneriere eine E-Mail für Social Selling auf LinkedIn:"
        generated_email = generate_linkedIn_email(generated_prompt)
        st.write(generated_email)
    else:
        st.error("Ein Fehler ist beim Abrufen einer oder mehrerer Webseiten aufgetreten. Bitte überprüfen Sie die URLs und versuchen Sie es erneut.")