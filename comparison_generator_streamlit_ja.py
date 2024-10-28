# Import dependencies    
import datetime    
import json    
import time    
import os    
import shutil
from openai import AzureOpenAI    
from dotenv import load_dotenv    
import copy    
import textwrap    
import requests    
import threading    
import streamlit as st    
from queue import Queue  
from PIL import Image
import base64
import io
import fitz  # PyMuPDF
import pandas as pd
from random import randint
from process_inputs import process_inputs  
# Load environment variables    
load_dotenv("./.env")    

# Define a flag to toggle deletion of the temporary folder
DELETE_TEMP_FOLDER = os.getenv("DELETE_TEMP_FOLDER", "true").lower() == "true"
TEMP_FOLDER = "./use-cases/Custom Scenario/images"

import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the value of the 'debug_mode' environment variable, default to 'true' if not set
offline_mode = os.getenv('offline_mode', 'true')
# Print the value
print(f"offline_mode: {offline_mode}")
# Log the value
logging.info(f"offline_mode: {offline_mode}")


offline_message="この機能はオフラインモードでは無効です。ローカルでこのアプリケーションをホストして独自のAPIキーで試してみてください。詳細については daka@microsoft.comにお問い合わせください。"

# Function to read XML file content as a string  
def load_use_case_from_file(file_path):  
    with open(file_path, 'r') as file:  
        return file.read()  
    
def get_csv_data(use_case,column_name):
    # Load the CSV file into a DataFrame
    csv_file_path = './o1-vs-4o-scenarios_ja.csv'
    df = pd.read_csv(csv_file_path)
    row = df[df['Use Case'] == use_case]
    if not row.empty:
        return row.iloc[0][column_name]
    else:
        return f"Error - {column_name} not found."

def save_csv_data(use_case, column_name, value):
    # Check if debug_mode is true
    if os.getenv('debug_mode') == 'true':
        # Load the CSV file into a DataFrame
        csv_file_path = './o1-vs-4o-scenarios_ja.csv'
        df = pd.read_csv(csv_file_path)
        
        # Check if the use case already exists
        if use_case in df['Use Case'].values:
            df.loc[df['Use Case'] == use_case, column_name] = value
            # Save the updated DataFrame back to the CSV file
            df.to_csv(csv_file_path, index=False)
        else:
            return f"Error - Use case '{use_case}' not found."


# Define function for calling GPT4o with streaming  
def gpt4o_call(system_message, user_message, result_dict, queue, selected_use_case):    
    if offline_mode == 'true':
        response_text = get_csv_data(selected_use_case, 'gpt4o')
        for i in range(0, len(response_text), 50):  # Simulate streaming
            queue.put(response_text[:i+50])
            time.sleep(0.2)
        result_dict['4o'] = {
            'response': response_text,
            'time': get_csv_data(selected_use_case, 'gpt4o_time')
        }
    else:
        client = AzureOpenAI(    
            api_version=os.getenv("4oAPI_VERSION"),    
            azure_endpoint=os.getenv("4oAZURE_ENDPOINT"),    
            api_key=os.getenv("4oAPI_KEY")    
        )    
        
        start_time = time.time()    
        
        completion = client.chat.completions.create(    
            model=os.getenv("4oMODEL"),    
            messages=[    
                {"role": "system", "content": system_message},    
                {"role": "user", "content": user_message},    
            ],    
            stream=True  # Enable streaming  
        )    
        
        response_text = ""  
        for chunk in completion:  
            if chunk.choices and chunk.choices[0].delta.content:  
                response_text += chunk.choices[0].delta.content  
                queue.put(response_text)  
        
        elapsed_time = time.time() - start_time    
        
        result_dict['4o'] = {    
            'response': response_text,    
            'time': elapsed_time    
        }    
        queue.put(f"Elapsed time: {elapsed_time:.2f} seconds")    

def o1_call(system_message, user_message):
    client = AzureOpenAI(    
        api_version=os.getenv("o1API_VERSION"),    
        azure_endpoint=os.getenv("o1AZURE_ENDPOINT"),    
        api_key=os.getenv("o1API_KEY")    
    )    
    
    start_time = time.time()    
    
    prompt = system_message + user_message

    completion = client.chat.completions.create(    
        model=os.getenv("o1API_MODEL"),    
        messages=[      
            {"role": "user", "content": prompt},    
        ],    
    )
    elapsed_time = time.time() - start_time   
    messageo1=completion.choices[0].message.content 
    return messageo1, elapsed_time  

# Define function for calling O1 and storing the result  
def o1_call_simultaneous_handler(system_message, user_message, result_dict,selected_use_case ):    
    if offline_mode == 'true':
        response = get_csv_data(selected_use_case, 'o1')
        # Sleep for the time taken by o1
        o1_time_elapsed = get_csv_data(selected_use_case, 'o1_time')
        time.sleep(o1_time_elapsed)
        print("SLEPT FOR ", o1_time_elapsed)
        result_dict['o1'] = {
            'response': response,
            'time': get_csv_data(selected_use_case, 'o1_time')
        }
    else:
        response,elapsed_time=o1_call(system_message, user_message)
        
        result_dict['o1'] = {    
            'response': response,    
            'time': elapsed_time    
        }    

# Define function for comparing responses using O1  
def compare_responses(response_4o, response_o1):  
    system_message = "You are an expert reviewer, who is helping review two candidates responses to a question. Please output the review in Japanese."  
    user_message = f"Compare the following two responses and summarize the key differences:\n\nResponse 1 GPT-4o Model:\n{response_4o}\n\nResponse 2 o1 Model:\n{response_o1}. Generate a succinct comparison, and call out the key elements that make one response better than another. Be critical in your analysis."  
    comparison_result, _ = o1_call(system_message, user_message)  

    return comparison_result  

# Define function for comparing responses using O1  
def compare_responses_simple(response_4o, response_o1):  
    system_message = "You are an expert reviewer, who is helping review two candidates responses to a question. Please output the review in Japanese."  
    user_message = f"Compare the following two responses and summarize the key differences:\n\nResponse 1 GPT-4o Model:\n{response_4o}\n\nResponse 2 o1 Model:\n{response_o1}. Generate a succinct comparison, and call out the key elements that make one response better than another. Be succinct- only use 3 sentences."  
    comparison_result, _ = o1_call(system_message, user_message)  
    
    return comparison_result  

# Function to process images and convert them to text using GPT-4o
def process_images(images):
    client = AzureOpenAI(
        api_version=os.getenv("4oAPI_VERSION"),
        azure_endpoint=os.getenv("4oAZURE_ENDPOINT"),
        api_key=os.getenv("4oAPI_KEY")
    )

    descriptions = []
    for image in images:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        system_prompt = "Generate a highly detailed text description of this image, making sure to capture all the information within the image as words. If there is text, tables or other text based information, include this in a section of your response as markdown."
        response = client.chat.completions.create(
            model=os.getenv("4oMODEL"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "Here is the input image:"},
                    {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{img_str}', "detail": "low"}}
                ]}
            ],
            temperature=0,
        )
        descriptions.append(response.choices[0].message.content)
    
    return descriptions

def process_pdf(pdf_path, output_folder):
    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{page_num + 1}.jepg")
        img.save(image_path, "JPEG")

def load_images_and_descriptions(selected_title):
    use_case_folder = f"./use-cases/{selected_title}/images"

    if os.path.exists(use_case_folder):
        image_files = [os.path.join(use_case_folder, f) for f in os.listdir(use_case_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        descriptions = []
        for img_file in image_files:
            base_name = os.path.splitext(os.path.basename(img_file))[0]
            description_path = os.path.join(use_case_folder, f"{base_name}.txt")
            if os.path.exists(description_path):
                with open(description_path, 'r', encoding='utf-8') as f:
                    description = f.read()
                    descriptions.append((img_file, description))
    else:
        image_files = []
        descriptions = []

    if descriptions:
        st.session_state['descriptions'] = descriptions
    else:
        st.session_state['descriptions'] = []

# Streamlit app    
def main():

    def render_images_and_descriptions():  
        cols = st.columns(3)  # Adjust the number of columns as needed  
        fixed_height = 200  # Fixed height for images in pixels  
    
        for i, (img_path, description) in enumerate(st.session_state['descriptions']):  
            image = Image.open(img_path)  
            col = cols[i % 3]  
            with col.container():  
                buffered = io.BytesIO()  
                image.save(buffered, format="JPEG")  
                img_str = base64.b64encode(buffered.getvalue()).decode()  
                # Generate a unique key based on description content  
                unique_id = f"{i}"  
                col.markdown(  
                    f"""  
                    <style>  
                        .image-container-{unique_id} {{  
                            height: {fixed_height}px;  
                            display: flex;  
                            align-items: center;  
                            justify-content: center;  
                            overflow: hidden;  
                            margin-bottom: 5px;  
                        }}  
                        .image-container-{unique_id} img {{  
                            height: 100%;  
                            width: auto;  
                            object-fit: cover;  
                        }}  
                    </style>  
                    <div class='image-container-{unique_id}'>  
                        <img src='data:image/jpeg;base64,{img_str}' alt='Image'>  
                    </div>  
                    """,  
                    unsafe_allow_html=True  
                )  
                # Use st.text_area for the description with a unique key  
                col.text_area("Description", description, height=100, key=f"description_{i}")  

    st.set_page_config(page_title="Azure OpenAI Studio 実験ラボ | GPT-4o vs 4o 比較ツール",page_icon="./favicon.ico", layout="wide")
    selected_item = st.sidebar.empty()

    def set_selected_item(item):
        st.session_state.selected_title = item
        load_images_and_descriptions(item)

    # Create two columns
    col1, col2 = st.sidebar.columns([8, 3])  # Adjust the ratio as needed

    # Place the title in the first column
    with col1:
        st.title("Azure OpenAI 実験ラボ 🔬")
    # Place the image in the second column
    with col2:
        st.text("")
        st.image("./azureopenaistudio.png", width=50) 
    st.sidebar.subheader("GPT-4o vs o1-preview 比較ツール") 
    st.sidebar.markdown("---")  

    # Custom CSS to make buttons full width and prevent wrapping
    st.sidebar.markdown("""
        <style>
        .stButton button {
            width: 100%;
            white-space: nowrap;
        }
        </style>
    """, unsafe_allow_html=True)

    # API key
    #st.sidebar.subheader("Azure OpenAI Service Configuration")
    #st.sidebar.text_input("Azure OpenAI o1 Key", type="password")
    #st.sidebar.text_input("Azure OpenAI o1 Endpoint")
    #st.sidebar.text_input("Azure OpenAI o1 Model Deployment Name") #gpt-4o-2024-08-06


    #st.sidebar.text_input("Azure OpenAI gpt-4o Key", type="password")
    #st.sidebar.text_input("Azure OpenAI gpt-4o Endpoint")
    #st.sidebar.text_input("Azure OpenAI gpt-4o Model Deployment Name")

    # Divider in the sidebar
    st.sidebar.markdown("---")

    # # Hidden text input to store the selected item
    # st.sidebar.text_input("", key="selected_item", on_change=lambda: set_selected_item(st.session_state.selected_item))
    if st.sidebar.button("Custom Scenario", key="custom_1"):
        if offline_mode == 'true':
            st.toast(offline_message, icon="⚠️")
        else:
            set_selected_item("Custom Scenario")
            print(offline_mode)
    st.sidebar.markdown("---") 

    
    # Insurance section
    st.sidebar.header("保険")  
    if st.sidebar.button("住宅保険請求 ⭐️", key="insurance_1"):
        set_selected_item("住宅保険請求")
    if st.sidebar.button("自動車保険請求 ⭐️", key="insurance_2"):
        set_selected_item("自動車保険請求")
    if st.sidebar.button("顧客サービスと関係保持", key="insurance_3"):
        set_selected_item("顧客サービスと関係保持")
    if st.sidebar.button("製品開発とイノベーション", key="insurance_4"):
        set_selected_item("製品開発とイノベーション")
    if st.sidebar.button("リスク管理とコンプライアンス", key="insurance_5"):
        set_selected_item("リスク管理とコンプライアンス")
    st.sidebar.markdown("---")  


    # Retail section
    st.sidebar.header("小売業")  
    if st.sidebar.button("在庫管理とサプライチェーン管理", key="retail_1"):
        set_selected_item("在庫管理とサプライチェーン管理")
    if st.sidebar.button("マーチャンダイジングと価格設定", key="retail_2"):
        set_selected_item("マーチャンダイジングと価格設定")
    if st.sidebar.button("顧客セグメンテーションとパーソナライゼーション", key="retail_3"):
        set_selected_item("顧客セグメンテーションとパーソナライゼーション")
    if st.sidebar.button("オムニチャネルとEコマース", key="retail_4"):
        set_selected_item("オムニチャネルとEコマース")
    if st.sidebar.button("ロイヤリティと顧客維持", key="retail_5"):
        set_selected_item("ロイヤリティと顧客維持")
    st.sidebar.markdown("---")

    # Telecommunications section
    st.sidebar.header("通信")  
    if st.sidebar.button("ネットワーク計画と最適化", key="telecom_1"):
        set_selected_item("ネットワーク計画と最適化")
    if st.sidebar.button("サービス開発とイノベーション", key="telecom_2"):
        set_selected_item("サービス開発とイノベーション")
    if st.sidebar.button("顧客獲得と維持", key="telecom_3"):
        set_selected_item("顧客獲得と維持")
    if st.sidebar.button("請求と収益管理", key="telecom_4"):
        set_selected_item("請求と収益管理")
    if st.sidebar.button("規制遵守と報告", key="telecom_5"):
        set_selected_item("規制遵守と報告")
    st.sidebar.markdown("---")

    # Utilities section
    st.sidebar.header("ユーティリティ")  
    if st.sidebar.button("需要と供給管理", key="utilities_1"):
        set_selected_item("需要と供給管理")
    if st.sidebar.button("資産とネットワーク管理", key="utilities_2"):
        set_selected_item("資産とネットワーク管理")
    if st.sidebar.button("顧客サービスと請求", key="utilities_3"):
        set_selected_item("顧客サービスと請求")
    if st.sidebar.button("エネルギー効率と持続可能性", key="utilities_4"):
        set_selected_item("エネルギー効率と持続可能性")
    if st.sidebar.button("規制遵守と報告", key="utilities_5"):
        set_selected_item("規制遵守と報告")
    st.sidebar.markdown("---")


    # Banking section
    st.sidebar.header("銀行")  
    if st.sidebar.button("信用リスク評価と管理 ⭐️", key="banking_1"):
        set_selected_item("信用リスク評価と管理")
    if st.sidebar.button("不正検知と防止", key="banking_2"):
        set_selected_item("不正検知と防止")
    if st.sidebar.button("規制遵守と報告", key="banking_3"):
        set_selected_item("規制遵守と報告")
    if st.sidebar.button("顧客関係管理", key="banking_4"):
        set_selected_item("顧客関係管理")
    if st.sidebar.button("投資とポートフォリオ管理", key="banking_5"):
        set_selected_item("投資とポートフォリオ管理")
    st.sidebar.markdown("---")


    # Mining section
    st.sidebar.header("鉱業")  
    if st.sidebar.button("探査と実現可能性", key="mining_1"):
        set_selected_item("探査と実現可能性")
    if st.sidebar.button("鉱山計画と設計", key="mining_2"):
        set_selected_item("鉱山計画と設計")
    if st.sidebar.button("生産と加工", key="mining_3"):
        set_selected_item("生産と加工")
    if st.sidebar.button("環境と社会的影響", key="mining_4"):
        set_selected_item("環境と社会的影響")
    if st.sidebar.button("健康と安全", key="mining_5"):
        set_selected_item("健康と安全")

    # Healthcare section
    st.sidebar.header("医療")  
    if st.sidebar.button("診断と治療", key="healthcare_1"):
        set_selected_item("診断と治療")
    if st.sidebar.button("ケアの調整と管理", key="healthcare_2"):
        set_selected_item("ケアの調整と管理")
    if st.sidebar.button("病気予防と健康促進", key="healthcare_3"):
        set_selected_item("病気予防と健康促進")
    if st.sidebar.button("研究とイノベーション", key="healthcare_4"):
        set_selected_item("研究とイノベーション")
    if st.sidebar.button("コンプライアンスと報告", key="healthcare_5"):
        set_selected_item("コンプライアンスと報告")
    st.sidebar.markdown("---")  

    # Education section
    st.sidebar.header("教育")  
    if st.sidebar.button("カリキュラム設計と提供", key="education_1"):
        set_selected_item("カリキュラム設計と提供")
    if st.sidebar.button("評価と査定", key="education_2"):
        set_selected_item("評価と査定")
    if st.sidebar.button("学生支援とエンゲージメント", key="education_3"):
        set_selected_item("学生支援とエンゲージメント")
    if st.sidebar.button("専門能力開発と協力", key="education_4"):
        set_selected_item("専門能力開発と協力")
    if st.sidebar.button("管理と運営", key="education_5"):
        set_selected_item("管理と運営")
    st.sidebar.markdown("---")
    
    if 'selected_title' not in st.session_state or not st.session_state['selected_title']:
        st.markdown("### 概要")
        st.markdown("このツールは、OpenAIのo1-previewモデルとGPT-4oモデルの違いを探るために設計されています。o1は、LLM（大規模言語モデル）向けの高度な推論機能を解放する新しいクラスのモデルです。問題について時間をかけて考えることで、o1はさまざまなエッジケースや状況を考慮し、より優れた結論に達することができます。ただし、その代償として待機時間が長くなります。\no1は多くの産業に変革をもたらす可能性があり、このツールを使ってその違いを探ることができます。")
        st.markdown("### 手順")
        st.markdown("左側のシナリオをクリックして開始してください。また、「カスタムシナリオ」を選択して独自のシナリオをアップロードすることもできます。")
        if offline_mode=='true':
            st.markdown("### ⚠️オフラインモード⚠️")
            st.markdown("このツールは現在オフラインモードで実行されています。GPT-4oおよびo1の動作を示すすべてのシナリオを探索し実行することは可能ですが、プロンプトの変更、ファイルのアップロード、独自のカスタムシナリオの追加はできません。ライブで試すには、独自のAPIキーでローカルにこのアプリケーションをホストしてください。詳細については、daka@microsoft.com までご連絡ください。")
        st.markdown("### 著作権表示")
        st.markdown("このデモに関する詳細情報については、Daiki Kanemitsu にお問い合わせください。プロダクト開発を統括したLuca Stamatescu、プロンプト戦略を開発したSalim Naim、および業界シナリオとユースケースを提供したIbrahim Hamzaに感謝します。")
    else:
        # Main content
        st.title(st.session_state.get("selected_title", "Custom Scenario"))


        # Custom CSS to hide Streamlit header and footer and adjust padding
        hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .css-18e3th9 {padding-top: 0;}
            .css-1d391kg {padding-top: 0;}
            </style>
        """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

        
        # Retrieve the selected use case from session state
        selected_use_case = st.session_state.get("selected_title", "Custom Scenario")



        st.markdown("##### 概要")
        overview = get_csv_data(selected_use_case,"Overview")

        st.markdown(overview)


        st.markdown("##### 詳細な内訳")
        # Get the default input based on the selected use case
        default_input = get_csv_data(selected_use_case,"Prompt")

        # Input box (takes up the width of the screen)   
        #  
        user_input = st.text_area("", value=default_input, height=200)    

        # Section to upload supporting documents
        st.markdown("##### サポートドキュメントのアップロード")
        
        # Use session state to store uploaded files and the uploader key
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = None
        if 'uploader_key' not in st.session_state:
            st.session_state.uploader_key = str(randint(1000, 100000000))
        
        # File uploader with a unique key
        uploaded_files = st.file_uploader("Choose images or PDFs", accept_multiple_files=True, type=["jpeg", "pdf"])
        
        # Button to delete uploaded files
        if st.button("Delete uploaded files"):
            if offline_mode == 'true':
                st.toast(offline_message, icon="⚠️")
            else:
                if DELETE_TEMP_FOLDER and os.path.exists(TEMP_FOLDER):
                    shutil.rmtree(TEMP_FOLDER)  
                    st.session_state.descriptions=None
            

        # Process uploaded files
        if uploaded_files and st.button("Upload Files"):
            if offline_mode == 'true':
                st.toast(offline_message, icon="⚠️")
            else:
                process_inputs(uploaded_files)
                load_images_and_descriptions("Custom Scenario")
    
        # Display images as tiles with descriptions  
        if 'descriptions' in st.session_state and st.session_state.descriptions!=None:
            # Call the function to render images and descriptions
            render_images_and_descriptions()
    
        # Add a checkbox to toggle the comparison
        compare_models = st.checkbox("o1-previewの出力のみ表示", value=False)
      
        # Button to submit    
        if st.button("Submit"): 
            with st.spinner('Processing...'):
                if st.session_state['descriptions']:  
                    # Ensure descriptions is a string
                    concatenated_descriptions=""
                    descriptions = st.session_state['descriptions']
                    if isinstance(descriptions, list):
                        for description in descriptions:
                            concatenated_descriptions=concatenated_descriptions+description[1]
                    st.session_state['prompt'] = user_input + "\n\n" + concatenated_descriptions
                else:  
                    st.session_state['prompt'] = user_input

                # Conditionally display columns based on the checkbox state
                if not compare_models:
                    col1, col2 = st.columns(2)    
                else:
                    col2 = st.container()
                
                if not compare_models:
                    with col1:    
                        st.subheader("4o Response")
                        st.markdown("---")
                        response_placeholder_4o = st.empty()  
                        st.markdown("---")
                        st.markdown("##### Timing")    
                        time_placeholder_4o = st.markdown("Processing...")   
        
                with col2:    
                    st.subheader("o1-preview Response")   
                    st.markdown("---")
                    response_placeholder_o1 = st.empty() 
                    st.markdown("---")
                    st.markdown("##### Timing")   
                    time_placeholder_o1 = st.markdown("Processing...")   
                
                # Dictionary to store results    
                result_dict = {}    
                queue = Queue()  
                
                # Start threads for both API calls    
                threads = []    
                t1 = threading.Thread(target=gpt4o_call, args=("You are a helpful AI assistant.", st.session_state['prompt'], result_dict, queue,selected_use_case))    
                t2 = threading.Thread(target=o1_call_simultaneous_handler, args=("You are a helpful AI assistant.", st.session_state['prompt'], result_dict,selected_use_case))    
                threads.append(t1)    
                threads.append(t2)    
                t1.start()    
                t2.start()    
                
                if not compare_models:
                    # Update the Streamlit UI with the streamed response  
                    while t1.is_alive():  
                        while not queue.empty():  
                            response_placeholder_4o.write(queue.get())  
                        time.sleep(0.1)  
                
                # Wait for both threads to complete    
                for t in threads:    
                    t.join()    
                
                # Display the 4o response and elapsed time  
                if not compare_models:
                    with col1:
                        response_placeholder_4o.write(result_dict['4o']['response'])  
                        time_placeholder_4o.write(f"Elapsed time: {result_dict['4o']['time']:.2f} seconds")  
                        if os.getenv('debug_mode') == 'true':
                            save_csv_data(selected_use_case, "gpt4o_time", float(round(result_dict['4o']['time'],2)))
                            save_csv_data(selected_use_case, "gpt4o", result_dict['4o']['response'])


                # Display the O1 response and elapsed time    
                with col2:    
                    response_placeholder_o1.write(result_dict['o1']['response'])   
                    time_placeholder_o1.write(f"Elapsed time: {result_dict['o1']['time']:.2f} seconds")    
                    if os.getenv('debug_mode') == 'true':
                        save_csv_data(selected_use_case, "o1_time", float(round(result_dict['o1']['time'],2)))
                        save_csv_data(selected_use_case, "o1", result_dict['o1']['response'])

            if not compare_models:
                st.markdown("---")
                # Compare the responses and display the comparison  
                st.subheader("Comparison of Responses - Overview")  

                with st.spinner('Processing...'):
                    if offline_mode == 'true':
                        comparison_result = get_csv_data(selected_use_case, 'simple_comparison')
                        # Simulate a wait time
                        time.sleep(15)
                    else:
                        comparison_result = compare_responses_simple(result_dict['4o']['response'], result_dict['o1']['response'])  
                    st.write(comparison_result)
                    save_csv_data(selected_use_case, "simple_comparison", comparison_result)

                st.markdown("---")
                # Compare the responses and display the comparison  
                st.subheader("Comparison of Responses - Detailed")  

                with st.spinner('Processing...'):
                    if offline_mode == 'true':
                        comparison_result = get_csv_data(selected_use_case, 'complex_comparison')
                        # Simulate a wait time
                        time.sleep(15)
                    else:
                        comparison_result = compare_responses(result_dict['4o']['response'], result_dict['o1']['response'])  
                    st.write(comparison_result)
                    save_csv_data(selected_use_case, "complex_comparison", comparison_result)



if __name__ == "__main__":
    main()