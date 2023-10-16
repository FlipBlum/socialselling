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

# 0. API Keys
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

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

# 2. Tool fürs scraping
def scrape_website(objective: str, url: str):
    # Webseite scrapen und den Inhalt zusammenfassen, basierend auf dem Zweck, wenn der Inhalt zu groß ist.
    # Der Zweck ist die ursprüngliche Aufgabe, die der Benutzer dem Agenten gegeben hat, und die URL ist die URL der zu scrapenden Website.

    print("Scraping website...")
    # Definiere die Header für die Anfrage
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Definiere die Daten, die in der Anfrage gesendet werden sollen
    data = {
        "url": url
    }

    # Konvertiere das Python-Objekt in einen JSON-String
    data_json = json.dumps(data)

    # Sende den POST request
    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Überprüfe den HTTP-Antwortstatuscode
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()

        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")

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
    Wir sind die Schlander & Blum GmbH. Wir verkaufen Qualitäts, Rework und Montage Outsourcingkapazitäten an Unternehmen aller Branchen. Bei Schlander & Blum können sie sich auf höchste Qualität verlassen, sei es mit Automatisierungstechnik, Sondermaschinenbau oder Menschen, wir liefern das was sie brauchen und mehr.

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

    # Schritt 2: Webseiten-Scraping
    company_content = scrape_website("Unternehmenszusammenfassung", company_url)
    company_linkedin_content = scrape_website("LinkedIn Unternehmenszusammenfassung", company_linkedin_url)
    person_content = scrape_website("Personenzusammenfassung", linkedin_url)

    if company_content is not None and company_linkedin_content is not None and person_content is not None:
        # Schritt 3: Textgenerierung mit OpenAI's ChatGPT
        company_summary = summary("Zusammenfassung des Unternehmens", company_content + company_linkedin_content)
        person_summary = summary("Zusammenfassung der Person", person_content)

        # Schritt 4: Anzeige der Zusammenfassungen
        st.subheader("Zusammenfassung zur Person:")
        st.write(person_summary)

        st.subheader("Zusammenfassung zum Unternehmen:")
        st.write(company_summary)

        # Anzeige der zusätzlichen Informationen
        st.subheader("Weitere Informationen:")
        st.write(weitere_informationen)

        st.subheader("Generierte E-Mail:")
        # Generiere die E-Mail mit dem erstellten Prompt
        generated_prompt = f"Company Summary: {company_summary}\nPerson Summary: {person_summary}\nAdditional Information: {weitere_informationen}\n\nGeneriere eine E-Mail für Social Selling auf LinkedIn:"
        generated_email = generate_linkedIn_email(generated_prompt)
        st.write(generated_email)
    else:
        st.error("Ein Fehler ist beim Abrufen einer oder mehrerer Webseiten aufgetreten. Bitte überprüfen Sie die URLs und versuchen Sie es erneut.")