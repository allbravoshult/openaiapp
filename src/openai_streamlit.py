# openai_streamlit.py
import ast
import base64
import io
import json
import os
import re
import streamlit as st
import pandas as pd
import time
from openai_utils import call_openai_api, display_costs, reset_cost_variables

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


def parse_api_response(response, row_index):
    try:
        content = response.choices[0].message.content

        # Remove Markdown code block syntax
        content = content.replace("```json\n", "").replace("\n```", "").strip()
        # Remove newline characters and extra spaces within the JSON string
        content = content.replace("\n", "")
        # Remove control characters from the JSON string
        content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)
        # Replace special double quotation marks with standard double quotation marks
        content = content.replace('“', '"').replace('”', '"')

        try:
            response_data = json.loads(content)
        except (SyntaxError, ValueError):
            # If the response is not a valid Python literal, try to fix it by adding double quotes around property names
            content = re.sub(r"(\w+):", r'"\1":', content)
            # Remove backslashes before underscores and any other escaped characters
            content = re.sub(r"\\(.)", r"\1", content)
            response_data = ast.literal_eval(content)

        return response_data

    except (SyntaxError, ValueError) as e:
        st.error(f"Error parsing API response for row {row_index + 1}: {e}")
        st.error(f"Response content: {content}")
        return None
    except AttributeError as e:
        if "object has no attribute 'message'" in str(e):
            raise ValueError("The API response does not contain the expected 'message' attribute.")
        else:
            raise e
    except Exception as e:
        st.error(f"Error parsing API response for row {row_index + 1}: {e}")
        return None


def process_rows(df, prompt, model):
    start_time = time.perf_counter()
    total_interactions = len(df)
    results = []
    log_area = st.empty()
    progress_bar = st.progress(0)

    for index, row in df.iterrows():
        try:
            progress = int(((index + 1) / total_interactions) * 100)
            progress_bar.progress(progress)
            elapsed_time = time.perf_counter() - start_time
            log_area.text(f"Processing row {index + 1}/{total_interactions} - Last row took {elapsed_time:0.2f} seconds.")

            prompt_parameters = row.to_dict()
            formatted_prompt = prompt.format(**prompt_parameters)
            messages = [
                {"role": "system", "content": "You are an Environmental Analyst."},
                {"role": "user", "content": formatted_prompt}
            ]
            response = call_openai_api(messages, model)

            response_data = parse_api_response(response, index)
            if response_data:
                new_row = row.tolist() + list(response_data.values())
                results.append(new_row)
            else:
                raise ValueError("Failed to parse API response.")

            elapsed_time = time.perf_counter() - start_time
            st.write(f"Processed row {index+1} in {elapsed_time:0.2f} seconds.")

        except Exception as e:
            st.error(f"Error processing row {index + 1}: A general error occurred - {e}")
            continue

    if results:
        # Use the keys from the last processed row's response_data
        results_df = pd.DataFrame(results, columns=list(df.columns) + list(response_data.keys()))
        display_costs(st)
    else:
        # If no rows were processed successfully, create an empty DataFrame with the original columns
        results_df = pd.DataFrame(columns=list(df.columns))

    return results_df


def main():
    st.title('Mútua - OpenAI API')

    if 'validated' not in st.session_state:
        st.session_state.validated = False

    if not st.session_state.validated:
        user_api_key = st.text_input("Enter your Token:", type="password")
        if st.button('Verify'):
            if user_api_key == OPENAI_API_KEY:
                st.success("Token validated successfully!")
                st.session_state.validated = True
                st.experimental_rerun()
            else:
                st.error("Invalid Token. Please try again.")
        st.stop()

    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        try:
            if uploaded_file.type == 'text/csv':
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
                data = pd.read_excel(uploaded_file)
        except Exception as e:
            st.write(f"Error reading file: {e}")
            return

        st.write(data)
        prompt = st.text_area("Enter your prompt here:")
        model = st.selectbox("Choose GPT model:", ("gpt-3.5-turbo", "gpt-4-turbo-preview"))

        col1, col2 = st.columns(2)

        with col1:
            start_row = st.number_input('Enter the start row:', min_value=1, value=1, step=1)

        with col2:
            total_rows = len(data)
            row_options = [1, 3, 5, 10, 25, 50, 100, 250, 500, 1000]
            valid_options = [opt for opt in row_options if opt < total_rows]
            valid_options.append('all')
            selected_option = st.selectbox("How many rows to run:", valid_options)

            if selected_option == 'all':
                num_rows_to_run = total_rows
            else:
                num_rows_to_run = selected_option

        adjusted_df = data.iloc[start_row - 1: start_row - 1 + num_rows_to_run]

        if st.button('Run Prompts'):

            reset_cost_variables()

            # Update the current_model variable based on the selected model
            current_model = model

            result_data = process_rows(adjusted_df, prompt, model)
            st.write("Processing completed. You can download the output file below.")

            csv = result_data.to_csv(index=False)
            b64_csv = base64.b64encode(csv.encode()).decode()
            href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="output.csv">Download CSV File</a>'
            st.markdown(href_csv, unsafe_allow_html=True)

            excel_io = io.BytesIO()
            result_data.to_excel(excel_io, index=False, engine='openpyxl')
            excel_b64 = base64.b64encode(excel_io.getvalue()).decode()
            href_excel = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{excel_b64}" download="output.xlsx">Download Excel File</a>'
            st.markdown(href_excel, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
