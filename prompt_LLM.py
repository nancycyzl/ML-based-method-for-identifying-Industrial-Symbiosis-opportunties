import os
import tqdm
import ollama
import argparse
import pandas as pd
from openai import OpenAI

from train import prepare_dataloaders
from utils import calculate_metrics

def make_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default="llama3.1", choices=["llama3.1", "gpt-4o-mini", "gpt-4o"])
    parser.add_argument('--EI_file', type=str, default="data/EI_data1_valid_need.csv", help="Path to the EI input csv file.")
    parser.add_argument('--output_dir', type=str, default="output_LLM", help="output directory for saving trained models")
    parser.add_argument('--output_file_id', type=int, default=0, help="output file id")
    parser.add_argument('--dataset_mode', type=str, default="produce", help="mixed / produce / need")
    parser.add_argument('--train_slice', type=float, default=1, help="Percentage of training data to test influence of data amount")
    parser.add_argument('--batch_size', type=int, default=256, help="batch size")

    args = parser.parse_args()
    return args


def prompt_llm(model, system_prompt, user_prompt):
    if "llama" in model.lower():
        response = ollama.chat(
            model=model,
            messages=[
              {
                "role": "system",
                "content": system_prompt,
              },
              {
                "role": "user",
                "content": user_prompt,
              }
            ])

        response_string = response['message']['content']
    elif "gpt" in model.lower():
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ])

        response_string = response.choices[0].message.content
    else:
        raise ("Model {} is not supported!".format(model))

    return response_string


def parse_llm_output(llm_response):
    # general format: Yes. XXXXXX.

    if "yes" in llm_response.lower():
        llm_result_binary = 1

    elif "no" in llm_response.lower():
        llm_result_binary = 0

    else:
        llm_result_binary = -1

    return llm_result_binary


def predict_with_llm(model, comp_description, res_description, mode="produce"):

    if mode == "produce":
        system_prompt = '''
Given a company activity description and a waste description.
You need to judge whether the company will generate this waste as output based on factual correctness.'''

        user_prompt = '''
Here is an example: 
activity description: paper and paperboard manufacturing
waste description: paper sludge
answer: Yes. Paper sludge is a waste from the activity.

Now please make judgement to the following activity and waste. Answer with Yes or No. Explain your answer within 30 words.
activity description: {}
waste description: {}
answer:
'''.format(comp_description, res_description)

    elif mode == "need":
        system_prompt = '''
Given a company activity description and a waste description.
You need to judge whether the company can reuse this waste as input based on factual correctness.'''

        user_prompt = '''
Here is an example: 
activity description: electricity production
waste description: food waste
answer: Yes. Food waste can be reused to produce biogas for electricity production.

Now please make judgement to the following activity and waste. Answer with Yes or No. Explain your answer within 30 words.
activity description: {}
waste description: {}
answer:
'''.format(comp_description, res_description)

    else:
        raise "Mode {} is not supported!".format(mode)

    llm_response = prompt_llm(model, system_prompt, user_prompt)
    llm_response_binary = parse_llm_output(llm_response)

    return llm_response_binary, llm_response


def load_test_data(args):
    # load the same test data as training ISClassifier
    train_loader, val_loader, test_loader = prepare_dataloaders(args)
    return test_loader


def main():
    args = make_argument()
    test_loader = load_test_data(args)

    # create output folder
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    all_predictions = []
    all_labels = []
    all_responses = []

    comp_description_list = []
    res_description_list = []

    # for company_descriptions, resource_descriptions, labels in test_loader:
    #     # one batch has args.batch_size data
    #     for company_description, resource_description, label in zip(company_descriptions, resource_descriptions, labels):

    for company_descriptions, resource_descriptions, labels in tqdm.tqdm(test_loader, desc="Processing Batches"):
        # Inner loop with progress bar
        for company_description, resource_description, label in tqdm.tqdm(
                zip(company_descriptions, resource_descriptions, labels),
                desc="Processing Items",
                leave=False,  # Ensures the inner progress bar is cleaned up after completion
                total=len(company_descriptions)  # Specify the total number of items in the batch
        ):
            result_binary, response_str = predict_with_llm(args.model, company_description, resource_description)

            all_predictions.append(result_binary)
            all_labels.append(label.item())
            all_responses.append(response_str)

            comp_description_list.append(company_description)
            res_description_list.append(resource_description)


    accuracy, precision, recall, f1 = calculate_metrics(all_labels, all_predictions)
    print("Prediction with {}: accuracy {}, precision {}, recall {}, f1 {}".format(args.model, accuracy, precision, recall, f1))

    # save metrics
    metrics_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [accuracy, precision, recall, f1]
    }
    metrics_df = pd.DataFrame(metrics_data)

    output_file = os.path.join(args.output_dir, "{}_{}_metrics_{}.xlsx".format(args.model, args.dataset_mode, args.output_file_id))
    metrics_df.to_excel(output_file, index=False)

    # save intermediate result
    data = {
        "Activity description": comp_description_list,
        "Waste description": res_description_list,
        "Responses": all_responses,
        "LLM result": all_predictions,
        "Label": all_labels
    }
    df = pd.DataFrame(data)

    output_file = os.path.join(args.output_dir, "{}_{}_results_{}.xlsx".format(args.model, args.dataset_mode, args.output_file_id))
    df.to_excel(output_file, index=False)

    print("Metrics and result files are saved at: {}".format(args.output_dir))


if __name__ == "__main__":
    main()
