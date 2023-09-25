import io
import streamlit as st
import PyPDF2
import yaml
import os
import time
import plotly.graph_objs as go
from streamlit_extras.colored_header import colored_header
from wordcloud import WordCloud
from streamlit_tags import st_tags
import matplotlib.pyplot as plt
from langchain.llms import OpenAIChat
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, PromptTemplate
import pandas as pd
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_lottie import st_lottie
st.set_page_config(page_title='Resume', layout='wide', page_icon="#")
with st.sidebar:
  st_lottie("https://lottie.host/d08859ec-b1b9-4c81-b0d3-a54b37de4485/tEZ00UNjDR.json")
  
def extract_text_from_binary(file):
    pdf_data = io.BytesIO(file)
    reader = PyPDF2.PdfReader(pdf_data)
    num_pages = len(reader.pages)
    text = ""

    for page in range(num_pages):
        current_page = reader.pages[page]
        text += current_page.extract_text()
    return text

def format_resume_to_yaml(resume):
    # Define your YAML template
    template = """
    Format the provided resume to this YAML template:
    ---
    Name: ''
    PhoneNumbers:
    - ''
    Websites:
    - ''
    Emails:
    - ''
    Achievements:
    - ''
    Projects:
    - ''
    Addresses:
    - street: ''
      city: ''
      state: ''
      zip: ''
      country: ''
    Summary: ''
    Education:
    - school: ''
      degree: ''
      fieldOfStudy: ''
      startDate: ''
      endDate: ''
    WorkExperience:
    - company: ''
      position: ''
      startDate: ''
      endDate: ''
    Skills:
    -  ''
    Certifications:
    -  ''
    {chat_history}
    {human_input}"""

    # Define the prompt
    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"],
        template=template
    )

    memory = ConversationBufferMemory(memory_key="chat_history")
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    llm_chain = LLMChain(
        llm=OpenAIChat(model="gpt-3.5-turbo",openai_api_key=openai_api_key),
        prompt=prompt,
        verbose=True,
        memory=memory,
    )

    # Predict using the LLM chain
    res = llm_chain.predict(human_input=resume)
    parsed_resume = yaml.safe_load(res)
    formatted_resume_df = pd.json_normalize(parsed_resume)

    return formatted_resume_df

# Streamlit UI

colored_header(
    label="Resume  Analyzer",
    description="This application is crafted to autonomously assess and appraise resumes submitted by job seekers.",
    color_name="red-70",
)

uploaded_file = st.file_uploader("Choose a Resume-PDF file", type="pdf")

if uploaded_file is not None:
    st.write("File uploaded successfully.")
    with st.spinner('Wait a moment...'):
                     time.sleep(130)
    
    common_data_science_skills = [
    'Python',
    'R',
    'SQL',
    'Machine Learning',
    'Data Analysis',
    'Data Visualization',
    'Statistics',
    'Deep Learning',
    'NLP','Data Wrangling','Aws','Azure','Powerbi','Tableau',
    'Big Data',
    'Hadoop',
    'Spark',
    'Tableau',
    'Data Mining','Computer Vision','Cloud Storage'
]
    resume_text = extract_text_from_binary(uploaded_file.read())

    # Format the resume to YAML
    formatted_resume_df = format_resume_to_yaml(resume_text)
    transposed_resume_df = formatted_resume_df.transpose().reset_index()
# Rename the columns to 'Details' and 'Values'
    transposed_resume_df.columns = ['Details', 'Values']
    st.markdown('__<p style="text-align:left; font-size: 20px; color: #1c0000">Formatted Resume :</P>__',
                unsafe_allow_html=True)
    st.dataframe(transposed_resume_df,use_container_width=True)
    add_vertical_space(3)

    work_experience_data = formatted_resume_df['WorkExperience'].iloc[0]

    companies = []
    positions = []

# Iterate through each dictionary entry and extract data
    for i, entry in enumerate(work_experience_data, 1):
     company = entry['company']
     position = entry['position']
     companies.append(f'Company{i}: {company}')
     positions.append(f'Position{i}: {position}')

# Create a Streamlit table to display the data
    st.markdown('__<p style="text-align:left; font-size: 20px; color: #1c0000">Work Experience :</P>__',
                unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({'Company': companies, 'Position': positions}),use_container_width=True)
    add_vertical_space(2)
    
    resume_skills = set([skill.lower() for skill in formatted_resume_df['Skills'][0]])
  
    resume_skills1 = [skill for skill in resume_skills]

    strengths = resume_skills

    areas_for_improvement = [skill for skill in common_data_science_skills if skill.lower() not in resume_skills]

    # Identify missing data science skills
    missing_skills = [skill for skill in common_data_science_skills if skill.lower() not in resume_skills]
    col1,col2 = st.columns([4,6])
    with col1:
     keywords = st_tags(label='Suggested Skills not in your Resume :',
                                   value=areas_for_improvement, key='1')
     keywords = st_tags(label='Skills in your Resume :',
                                   value=resume_skills1, key='2')
     #st.warning("Suggested Skills not in your Resume :")
     #st.table(missing_skills)
     add_vertical_space(3)

# Generate the summary report
    summary_report = {
    "Strengths": list(strengths),
    "Areas_for_improvement": areas_for_improvement
}
    max_length = max(len(values) for values in summary_report.values())

# Fill lists to make them of equal length
    summary_report = {k: v + [''] * (max_length - len(v)) for k, v in summary_report.items()}

# Create a DataFrame
    summary_df = pd.DataFrame(summary_report)

# Transpose the DataFrame
    #summary_df_transposed = summary_df.transpose()
    
# Display the summary report
    #st.markdown('__<p style="text-align:left; font-size: 20px; color: #1c0000">Summary Report :</P>__',
                #unsafe_allow_html=True)
    #st.table(summary_df)
    data = {
    'Topic': [
        "Educational Background",
        "Technical Skills",
        "Data Handling and Analysis",
        "Machine Learning and AI",
        "Data Storytelling and Communication",
        "Domain Knowledge",
        "Problem-Solving and Innovation",
        "Project Experience",
        "Research and Publications (for research-oriented roles)",
        "Continuous Learning",
        "Soft Skills",
    ],
    'Common Expectations/Qualifications': [
        "Bachelor's or advanced degree in related field",
        "Programming proficiency in Python or R; knowledge of data manipulation libraries and machine learning frameworks",
        "Data collection, cleaning, preprocessing, and statistical analysis; feature engineering",
        "Building and deploying machine learning models; deep learning knowledge for specific roles",
        "Effective communication skills; data visualization",
        "Familiarity with industry/domain; understanding of specific challenges",
        "Strong problem-solving and creative thinking",
        "Experience completing data science or AI projects; collaboration skills",
        "Research publications for research-oriented roles",
        "Staying updated with the latest developments",
        "Analytical thinking, adaptability, teamwork, time management",
    ]
}

# Create a DataFrame from the data
    df = pd.DataFrame(data)
    headerColor = 'grey'
    rowEvenColor = 'lightgrey'
    rowOddColor = 'white'
# Create a Plotly table
    table = go.Figure(data=[go.Table(
    header=dict(values=["Topic", "Common Expectations/Qualifications"],line_color='darkslategray',
                fill_color=headerColor,align='center',font=dict(color='white', size=12)),
    cells=dict(values=[df['Topic'], df['Common Expectations/Qualifications']],line_color='darkslategray',font = dict(color = 'darkslategray', size = 10),
            fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor, rowEvenColor,rowOddColor, rowEvenColor,rowOddColor,rowEvenColor,rowOddColor]*11],align='left'))
])

# Set the table layout
    table.update_layout(title="Expectations and Qualifications in Data Science :")

# Streamlit
    st.plotly_chart(table,use_container_width=True)
    
    strengths_text = ' '.join(list(strengths))
    wordcloud_strengths = WordCloud(width=600, height=300, background_color='white').generate(strengths_text)

# Generate word cloud for areas for improvement
    areas_for_improvement_text = ' '.join(areas_for_improvement)
    wordcloud_areas_for_improvement = WordCloud(width=600, height=300, background_color='white').generate(areas_for_improvement_text)


    col3,col4 = st.columns([5,5])
# Display word cloud for strengths
    with col3:
     st.markdown('__<p style="text-align:left; font-size: 20px; color: #1c0000">Strengths :</P>__',
                unsafe_allow_html=True)
     st.image(wordcloud_strengths.to_array(), use_column_width=True)
    with col4:
# Display word cloud for areas for improvement
     st.markdown('__<p style="text-align:left; font-size: 20px; color: #1c0000">Areas for Improvement :</P>__',
                unsafe_allow_html=True) 
     st.image(wordcloud_areas_for_improvement.to_array(), use_column_width=True)
     add_vertical_space(2)
     
     message = """
            Stay Updated : Keep up with the latest advancements, tools, and methodologies in data science by reading books,
            academic papers, online blogs, and following reputable sources.

            Machine Learning Concepts : Gain a deep understanding of fundamental machine learning concepts,
            algorithms, and techniques, including regression, clustering, classification, ensemble methods, and deep learning.

            Dealing with Real Data Challenges : Real-world data is often messy, incomplete, and inconsistent. Real-time projects help individuals learn how to clean, 
            preprocess, and handle such data, which is a critical aspect of real-world data science work.
           """

# Split the message into paragraphs
    paragraphs = message.split('\n\n')

# Display each paragraph within a colored container
    for idx, paragraph in enumerate(paragraphs):
    # Define a unique CSS class for each paragraph container
     class_name = f"paragraph-container-{idx}"

    # Apply a background color based on the paragraph number
     if idx % 3 == 0:
        container_style = "background-color: lightblue; padding: 10px; border-radius: 5px;"
     elif idx % 3 == 1:
        container_style = "background-color: lightgreen; padding: 10px; border-radius: 5px;"
     else:
        container_style = "background-color: lightcoral; padding: 10px; border-radius: 5px;"

    # Display the paragraph within the colored container
     add_vertical_space(2)
     st.markdown(
        f'<div class="{class_name}" style="{container_style}">\
            <b> prerequisites {idx + 1}:</b>\
            <p>{paragraph}</p>\
        </div>',
        unsafe_allow_html=True
    )
    
    def add_details(skill, details):
      return f"{skill}:\n{details}\n\n"

# Initialize the summary report
    summary_report = ""

# Check if "Machine Learning" is a missing skill
    if "Machine Learning" in areas_for_improvement:
    # Add details for "Machine Learning"
      ml_details = (
        "Consider enhancing your understanding and practical skills in Machine Learning. "
        "Explore topics such as supervised learning, unsupervised learning, "
        "and model evaluation. Familiarize yourself with popular ML libraries like "
        "scikit-learn and TensorFlow/Keras for effective implementation."
    )
      summary_report += add_details("Machine Learning", ml_details)

# Check if "Deep Learning" is a missing skill
    if "Deep Learning" in areas_for_improvement:
    # Add details for "Deep Learning"
      dl_details = (
        "To improve your resume, focus on Deep Learning concepts and applications. "
        "Deepen your understanding of neural networks, convolutional neural networks (CNNs), "
        "recurrent neural networks (RNNs), and popular deep learning frameworks like TensorFlow "
        "and PyTorch. Explore projects and courses related to deep learning for hands-on experience."
    )
      summary_report += add_details("Deep Learning", dl_details)
      
    if "Data Analysis" in areas_for_improvement:
    # Add details for "Data Analysis"
      data_analysis_details = (
        "Enhance your data analysis skills by practicing data wrangling, data cleaning, and "
        "statistical analysis. Learn to work with various data formats and use libraries like "
        "Pandas and NumPy for efficient data manipulation and analysis."
    )
      summary_report += add_details("Data Analysis", data_analysis_details)
      
    if "Computer Vision" in areas_for_improvement:  
     cv_details =  (
        "For a well-rounded skill set, delve into Computer Vision. Study algorithms and techniques "
        "used for image and video analysis. Familiarize yourself with Convolutional Neural Networks (CNNs), "
        "object detection, image segmentation, and image classification. Work on projects related to computer vision."
    )
     summary_report += add_details("Computer Vision", cv_details)
     
    if "NLP" in areas_for_improvement:  
     nlp_details =  (
        "Expand your skill set to include Natural Language Processing (NLP). Study methods and techniques for "
        "processing and analyzing text data. Learn about sentiment analysis, named entity recognition, machine "
        "translation, and topic modeling. Experiment with NLP libraries like NLTK, spaCy, and transformers."
     )
     summary_report += add_details("NLP", nlp_details) 
 
# Check if "Data Visualization" is a missing skill
    if "Data Visualization" in areas_for_improvement:
    # Add details for "Data Visualization"
      data_viz_details = (
        "Improve your data visualization skills by learning to present data in an effective and "
        "engaging manner. Explore popular visualization libraries such as Matplotlib, Seaborn, and "
        "Plotly. Practice creating various types of visualizations to convey insights from data."
    )
      summary_report += add_details("Data Visualization", data_viz_details)
      
    if "Cloud Storage"  in areas_for_improvement: 
      cloud_details = (
        "Familiarize yourself with cloud storage solutions such as Amazon S3, Google Cloud Storage, or Azure "
        "Blob Storage. Learn how to store and manage data in the cloud efficiently. Understand concepts like buckets, "
        "objects, access control, and cost optimization related to cloud storage."
    )
      summary_report += add_details("Cloud Storage",cloud_details)

# Display the summary report with detailed paragraphs
    with col2:
     st.subheader("Detailed Areas for Improvement :")
     st.write(summary_report)
     
