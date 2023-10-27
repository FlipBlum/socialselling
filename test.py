import os
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import json
import streamlit as st
from bs4 import BeautifulSoup

# 0. API Keys
serper_api_key = os.getenv("SERP_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# 1. Tool for the Search
def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers, data=payload)
    return response.text

# 2. Tool for Scraping
def scrape_website(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text()
        return text
    else:
        st.error(f"Failed to scrape the website. Status Code: {response.status_code}")
        return None

# 3. Tool for Summarizing
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

# 4. Email Generation
def generate_linkedIn_email(company_summary, person_summary, weitere_informationen):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
    prompt = f"""
    **Instructions:**
    - We are Schlander & Blum GmbH. We sell quality rework and assembly outsourcing capacities to companies in all industries. At Schlander & Blum, you can rely on the highest quality, whether with automation technology, special machine construction or people, we deliver what you need and more.

    **Company Summary:**
    {company_summary}

    **Person Summary:**
    {person_summary}

    **Additional Information:**
    {weitere_informationen}

    **Generate an email for Social Selling on LinkedIn:**
    """
    generated_email = llm.generate(prompt, max_tokens=200)
    return generated_email

# 5. Streamlit User Interface
st.title("LinkedIn Scraper & LLM Text Generation")

weitere_informationen = st.text_input("Additional Information")
company_url = st.text_input("Company URL")
company_linkedin_url = st.text_input("LinkedIn URL (Company)")
linkedin_url = st.text_input("LinkedIn URL (Person)")

if st.button("Start"):
    # Step 1: Search Queries
    company_search = search("site:linkedin.com " + company_url)
    company_linkedin_search = search("site:linkedin.com/company " + company_linkedin_url)

    # Step 2: Website Scraping
    company_content = scrape_website(company_url)
    company_linkedin_content = scrape_website(company_linkedin_url)
    person_content = scrape_website(linkedin_url)

    if company_content and company_linkedin_content and person_content:
        # Step 3: Text Generation with OpenAI's ChatGPT
        company_summary = summary("Company Summary", company_content + company_linkedin_content)
        person_summary = summary("Person Summary", person_content)

        # Step 4: Display Summaries
        st.subheader("Person Summary:")
        st.write(person_summary)

        st.subheader("Company Summary:")
        st.write(company_summary)

        # Display Additional Information
        st.subheader("Additional Information:")
        st.write(weitere_informationen)

        # Generate and Display Email
        st.subheader("Generated Email:")
        generated_email = generate_linkedIn_email(company_summary, person_summary, weitere_informationen)
        st.write(generated_email)
    else:
        st.error("An error occurred while retrieving one or more web pages. Please check the URLs and try again.")
