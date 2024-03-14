# openai_utils.py
import time
from openai import OpenAI, APIError, RateLimitError, AuthenticationError

# Global variables to track usage and costs
total_input_tokens = 0
total_output_tokens = 0
total_cost = 0.0
num_api_calls = 0

last_input_tokens = 0
last_output_tokens = 0
last_cost = 0.0

# Updated model costs to include separate costs for input and output tokens
MODELS_COSTS = {
    "gpt-4-turbo-preview": {
        "input": 0.01 / 1000,  # $0.01 per 1000 input tokens
        "output": 0.03 / 1000  # $0.03 per 1000 output tokens
    },
    "gpt-3.5-turbo-0125": {
        "input": 0.0005 / 1000,  # $0.0015 per 1000 input tokens
        "output": 0.0015 / 1000  # $0.002 per 1000 output tokens
    },
    "gpt-4": {
        "input": 0.03 / 1000,  # $0.03 per 1000 input tokens
        "output": 0.06 / 1000  # $0.06 per 1000 output tokens
    },
    "gpt-3.5-turbo": {
        "input": 0.001 / 1000,  # $0.0015 per 1000 input tokens
        "output": 0.002 / 1000  # $0.002 per 1000 output tokens
    }
}

# Default model
current_model = "gpt-4-turbo-preview"  # This can be changed based on the model you intend to use more frequently


def call_openai_api(messages, model=current_model, response_format=None, retry=False):
    global total_input_tokens, total_output_tokens, total_cost, num_api_calls, current_model
    global last_input_tokens, last_output_tokens, last_cost

    current_model = model

    if model not in MODELS_COSTS:
        available_models = ", ".join(MODELS_COSTS.keys())
        raise ValueError(f"Invalid model '{model}'. Available models are: {available_models}")

    try:
        # Create an OpenAI client instance
        client = OpenAI()

        # Make the API call with additional format parameter if provided
        if response_format:
            response = client.chat.completions.create(model=model, messages=messages, response_format=response_format)
        else:
            response = client.chat.completions.create(model=model, messages=messages)

        # Retrieve the tokens count
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        # Before updating global counters, store the last state
        last_input_tokens = input_tokens
        last_output_tokens = output_tokens
        last_cost = input_tokens * MODELS_COSTS[model]["input"] + output_tokens * MODELS_COSTS[model]["output"]

        # Update global counters
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        num_api_calls += 1

        # Calculate the costs separately for input and output
        input_cost = input_tokens * MODELS_COSTS[model]["input"]
        output_cost = output_tokens * MODELS_COSTS[model]["output"]

        # Update the total cost
        total_cost += (input_cost + output_cost)

        return response

    except AuthenticationError:
        print("Authentication failed. Check your API key.")
    except RateLimitError:
        print("Rate limit exceeded. Try again later.")
    except APIError as e:
        print(f"API error occurred: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

        if not retry:
            print("Retrying in 60 seconds...")
            time.sleep(1)
            return call_openai_api(messages, model, response_format=response_format, retry=True)
        else:
            print("Second attempt failed. No more retries.")
            return None


def display_costs(st):
    total_cost_str = f"Total Cost: $ {total_cost:.2f}"
    cost_per_row = total_cost / num_api_calls if num_api_calls > 0 else 0
    cost_per_row_str = f"Cost per Row: $ {cost_per_row:.2f}"
    estimated_cost_per_1000 = (total_cost / num_api_calls) * 1000 if num_api_calls > 0 else 0
    estimated_cost_per_1000_str = f"Estimated Cost per 1000 Rows: $ {estimated_cost_per_1000:.2f}"

    st.write(total_cost_str)
    st.write(cost_per_row_str)
    st.write(estimated_cost_per_1000_str)


def reset_cost_variables():
    global total_input_tokens, total_output_tokens, total_cost, num_api_calls
    global last_input_tokens, last_output_tokens, last_cost

    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    num_api_calls = 0

    last_input_tokens = 0
    last_output_tokens = 0
    last_cost = 0.0