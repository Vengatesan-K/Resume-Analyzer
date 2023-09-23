import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import time
from selenium.webdriver.chrome.service import Service
import pandas as pd
import re
import plotly.express as px
from streamlit_extras.colored_header import colored_header
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from streamlit_lottie import st_lottie
st.set_page_config(page_title='Resume', layout='wide', page_icon="#")
with st.sidebar:
  st_lottie("https://lottie.host/ae7f655c-7966-4ffc-bd7c-c6e3950c5b14/7U1y6mcLdr.json")
colored_header(
    label="LinkedIn Job Search",
    description="Select job location and keywords:",
    color_name="red-70",
)

location = st.text_input("Country",'India')  # Add more locations as needed
job_keywords = st.text_input("Job Keywords", "Data Scientist")

@st.experimental_singleton
def get_driver():
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

st.title("LinkedIn Job Scraper")
st.write("Select job location and keywords:")

location = st.text_input("Country")  # Add more locations as needed
job_keywords = st.text_input("Job Keywords", "Marketing Data Analysis")

if st.button("Scrape Jobs"):
    url = f'https://www.linkedin.com/jobs/search?keywords={job_keywords}&location={location}&trk=public_jobs_jobs-search-bar_search-submit'
    
    options = Options()
    options.add_argument('--disable-gpu')
    options.add_argument('--headless')
    
    with get_driver() as driver:
        driver.get(url)
        time.sleep(5)
        
        job_count_elements = driver.find_elements("css selector", ".results-context-header__job-count")
        if job_count_elements:
            y = job_count_elements[0].text
            y = re.sub(r'[^\d]', '', y)
            n = pd.to_numeric(y)
            
            data = []  # Initialize a list to store job data
            
            try:
                for i in range(n):
                    company = driver.find_elements("css selector", '.base-search-card__subtitle')[i].text
                    title = driver.find_elements("css selector", '.base-search-card__title')[i].text
                    
                    # The city can be in the same element or nearby, adjust the selector accordingly
                    city_element = driver.find_elements("css selector", '.job-search-card__location')[i].text
                    
                    # Append job data to the list
                    data.append({
                        'company': company,
                        'title': title,
                        'city': city_element
                    })
            except IndexError:
                print("no")
        
            # Create DataFrame from the collected job data
            job_data = pd.DataFrame(data)
            
            st.dataframe(job_data)
        else:
            st.write("No job count found. Check if the page loaded correctly.")
        
        city_counts = job_data['city'].value_counts()

        cmap = plt.get_cmap('viridis', len(city_counts))
        city_colors = [plt.cm.colors.rgb2hex(cmap(i)[:3]) for i in range(len(city_counts))]

        fig = px.bar(x=city_counts.index, y=city_counts.values, color=city_counts.index,
             color_discrete_map={city: color for city, color in zip(city_counts.index, city_colors)},
             labels={'x': 'City', 'y': 'Number of Jobs'}, title='Job Distribution by City')

        fig.update_layout(xaxis_tickangle=-45)

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No job count found. Check if the page loaded correctly.")
    
    driver.quit()
